from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

def calculate_match_score(user, opportunity):
    # 1. Match de Habilidades (NLP / TF-IDF)
    user_skills_text = " ".join(user.get('habilidades', []))
    opp_text = f"{opportunity.get('title', '')} {' '.join(opportunity.get('categories', []))} {opportunity.get('description', '')}"

    vectorizer = TfidfVectorizer().fit_transform([user_skills_text, opp_text])
    vectors = vectorizer.toarray()
    
    # Cosine Similarity entre o vetor do usuário e o da vaga
    skill_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # 2. Match de Localização
    location_score = 0.0
    if user.get('cidade') and opportunity.get('cidade'):
        if user['cidade'].strip().lower() == opportunity['cidade'].strip().lower():
            location_score = 1.0
        elif user.get('estado') == opportunity.get('estado'):
            location_score = 0.5

    # 3. Score Final Ponderado (70% Habilidades / Tags + 30% Localização)
    final_score = (skill_score * 0.7) + (location_score * 0.3)
    return float(final_score)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user = data.get('user', {})
    opportunities = data.get('opportunities', [])

    results = []
    for opp in opportunities:
        score = calculate_match_score(user, opp)
        results.append({
            'opportunityId': opp['id'],
            'score': round(score, 4)
        })

    # Ordena do maior match para o menor
    results.sort(key=lambda x: x['score'], reverse=True)

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)