from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone
import math
import re

app = Flask(__name__)

# Stopwords básicas em português para não poluir o TF-IDF
PT_STOPWORDS = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não',
    'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas',
    'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já'
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\sáàâãéèêíïóôõöúçñ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def calculate_age(nascimento: str) -> int | None:
    if not nascimento:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            dob = datetime.strptime(nascimento[:10], fmt)
            today = datetime.now()
            return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            continue
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_location_score(user: dict, opp: dict) -> float:
    modality = (opp.get('modality') or 'PRESENCIAL').upper()
    if modality == 'REMOTO':
        return 1.0

    # 1. Tenta distância real por coordenadas (se enviadas)
    u_lat, u_lng = user.get('latitude'), user.get('longitude')
    o_lat, o_lng = opp.get('latitude'), opp.get('longitude')
    if all(v is not None for v in (u_lat, u_lng, o_lat, o_lng)):
        dist_km = haversine_km(float(u_lat), float(u_lng), float(o_lat), float(o_lng))
        if dist_km <= 5:
            base_score = 1.0
        elif dist_km <= 20:
            base_score = max(0.4, 1.0 - (dist_km - 5) / 25.0)
        elif dist_km <= 50:
            base_score = 0.25
        else:
            base_score = 0.05
        return min(1.0, base_score + (0.15 if modality == 'HIBRIDO' else 0.0))

    # 2. Fallback por Cidade / Estado
    u_city = (user.get('cidade') or '').strip().lower()
    o_city = (opp.get('cidade') or '').strip().lower()
    if u_city and o_city and u_city == o_city:
        return 1.0

    u_state = (user.get('estado') or '').strip().upper()
    o_state = (opp.get('estado') or '').strip().upper()
    if u_state and o_state and u_state == o_state:
        return 0.65 if modality == 'HIBRIDO' else 0.45

    return 0.4 if modality == 'HIBRIDO' else 0.0


def compute_urgency_score(event_date_str: str) -> float:
    if not event_date_str:
        return 0.5
    try:
        event_dt = datetime.fromisoformat(event_date_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        days_until = (event_dt - now).total_seconds() / 86400.0
        if days_until < 0:
            return 0.0
        if days_until <= 7:
            return 1.0
        if days_until <= 30:
            return 0.75
        return 0.4
    except ValueError:
        return 0.5


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json() or {}
    user = data.get('user', {})
    opportunities = data.get('opportunities', [])

    if not opportunities:
        return jsonify([])

    user_age = calculate_age(user.get('nascimento'))
    user_skills = [s.strip().lower() for s in user.get('habilidades', []) if s]
    user_bio = user.get('bio', '')

    # Filtra apenas oportunidades elegíveis (ATIVA e idade mínima compatível)
    eligible_opps = []
    for opp in opportunities:
        if opp.get('status', 'ATIVA') != 'ATIVA':
            continue
        min_age = opp.get('minAge')
        if min_age is not None and user_age is not None and user_age < int(min_age):
            continue
        eligible_opps.append(opp)

    if not eligible_opps:
        return jsonify([])

    # Monta corpus completo para TF-IDF com peso extra em requiredSkills e categories
    user_doc = clean_text(f"{' '.join(user_skills * 2)} {user_bio}")
    opp_docs = []
    for opp in eligible_opps:
        req_skills = " ".join(opp.get('requiredSkills', []))
        categories = " ".join(opp.get('categories', []))
        title = opp.get('title', '')
        desc = opp.get('description', '')
        # Repete requiredSkills e categories para dar maior relevância semântica
        doc = clean_text(f"{title} {req_skills} {req_skills} {categories} {categories} {desc}")
        opp_docs.append(doc)

    # Calcula similaridade TF-IDF no corpus inteiro
    if user_doc and any(opp_docs):
        vectorizer = TfidfVectorizer(stop_words=PT_STOPWORDS, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([user_doc] + opp_docs)
        cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    else:
        cosine_scores = [0.0] * len(eligible_opps)

    results = []
    user_skills_set = set(user_skills)

    for idx, opp in enumerate(eligible_opps):
        tfidf_score = float(cosine_scores[idx])

        # Bônus direto por match exato em requiredSkills
        req_skills_set = {s.strip().lower() for s in opp.get('requiredSkills', []) if s}
        if req_skills_set and user_skills_set:
            exact_ratio = len(user_skills_set.intersection(req_skills_set)) / len(req_skills_set)
            skill_score = min(1.0, (tfidf_score * 0.6) + (exact_ratio * 0.4))
        else:
            skill_score = tfidf_score

        location_score = compute_location_score(user, opp)
        urgency_score = compute_urgency_score(opp.get('eventDate'))

        # Peso final: 55% Habilidades/Tags + 35% Localização/Modalidade + 10% Urgência do Evento
        final_score = (skill_score * 0.55) + (location_score * 0.35) + (urgency_score * 0.10)

        results.append({
            'opportunityId': opp['id'],
            'score': round(final_score, 4),
            'breakdown': {
                'skills': round(skill_score, 4),
                'location': round(location_score, 4),
                'urgency': round(urgency_score, 4)
            }
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)