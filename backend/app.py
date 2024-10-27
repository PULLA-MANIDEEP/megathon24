from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pymongo
from transformers import pipeline
import spacy
from collections import Counter
import re

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mental_health_db"]
analysis_collection = db["analyses"]

# Load pre-trained models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
nlp = spacy.load("en_core_web_sm")

# Mental health concern categories with keywords
concern_categories = {
   "Anxiety": [
        "anxiety", "anxious", "nervous", "worry", "worried", "panic", 
        "fearful", "jittery", "uneasy", "fidgety", "restless", "tense", 
        "overthinking", "stress", "hypervigilant", "edgy", "apprehensive", 
        "jumpy", "on edge", "troubled", "dread", "fret", "tremble", 
        "fidget", "twitch", "distress", "anticipation", "agitated", 
        "frantic", "overwhelmed", "fear of failure", "self-doubt", 
        "worried sick", "chest tightness", "mind racing", "panic attack", 
        "facing fears", "sweating", "short of breath", "feeling small", 
        "paranoia", "feeling trapped", "constant worrying", "feeling shaky", 
        "tense muscles", "feeling jumpy", "excessive worry", "fear of change",
        "avoiding situations", "nervous habits", "physical tension", "social anxiety",
        "post-traumatic stress", "feeling overwhelmed by uncertainty", "excessive caution"
    ],
    "Depression": [
        "depressed", "low", "sad", "hopeless", "worthless", "down", 
        "blue", "despair", "melancholy", "emptiness", "disinterest", 
        "unmotivated", "listless", "gloomy", "miserable", "flat", 
        "bummed", "disheartened", "weary", "heavy-hearted", "burdened", 
        "dim", "dark thoughts", "lack of joy", "feeling down", 
        "emotional pain", "despondent", "feeling empty", "grief", 
        "sorrow", "loneliness", "feeling isolated", "feeling inadequate", 
        "self-pity", "feeling heavy", "sadness", "loss of interest", 
        "tears", "crying", "mental fatigue", "self-loathing", 
        "worthlessness", "helplessness", "lack of energy", "disappointment",
        "feeling trapped in sadness", "unfulfilled", "dysthymia", "feeling like a burden",
        "anhedonia", "chronic sadness", "emptiness", "lost motivation", 
        "self-hate", "negative self-talk", "overwhelmed by life", "feeling invisible"
    ],
    "Stress": [
        "stress", "stressed", "overwhelmed", "pressure", "tense", 
        "burnout", "frantic", "strained", "fatigue", "nervous breakdown", 
        "exhausted", "strained", "juggling too much", "under pressure", 
        "overloaded", "high tension", "frenzied", "distressed", 
        "heavy load", "drained", "worn out", "trapped", "tight", 
        "feeling stretched", "feeling burdened", "demanding", 
        "chaotic", "too many responsibilities", "mental fatigue", 
        "physical strain", "feeling pulled in different directions", 
        "feeling frantic", "mind racing", "lack of balance", 
        "unable to relax", "overcommitment", "impatient", 
        "frustration", "time pressure", "feeling overwhelmed by tasks",
        "excessive demands", "lack of support", "coping mechanisms", 
        "need for downtime", "high expectations", "performance pressure"
    ],
    "Insomnia": [
        "sleep", "insomnia", "awake", "tired", "restless", "sleepless", 
        "fatigued", "dozing", "trouble sleeping", "nightmares", "tossing", 
        "turning", "sleep-deprived", "exhausted", "groggy", "sluggish", 
        "drowsy", "inability to relax", "staying up", "lost in thought", 
        "waking up too early", "not enough sleep", "sleeping pills", 
        "chronic insomnia", "irregular sleep", "sleep issues", 
        "uncomfortable", "sleep anxiety", "dream disturbances", 
        "poor sleep quality", "sleep disorders", "excessive wakefulness", 
        "troubled sleep", "head spinning", "mind racing at night", 
        "constant fatigue", "daytime drowsiness", "cognitive fog", 
        "restless legs", "tension headaches", "not feeling rested", 
        "racing thoughts at night", "emotional exhaustion"
    ],
    "Fear": [
        "fear", "afraid", "scared", "terrified", "panic", "dread", 
        "apprehensive", "alarm", "phobia", "fearful thoughts", 
        "anxiety", "timid", "frightened", "shocked", "horrified", 
        "apprehension", "intimidated", "threatened", "vulnerable", 
        "worried", "concerned", "troubled", "suspicious", "foreboding", 
        "paranoia", "fearing the worst", "irrational fears", 
        "heightened sensitivity", "specific fears", "claustrophobia", 
        "agoraphobia", "social anxiety", "fear of judgment", 
        "fear of rejection", "fear of failure", "feeling unsafe", 
        "anxiety attacks", "survivor's guilt", "feelings of impending doom",
        "trembling", "heart racing", "panic attacks", "sense of danger"
    ],
    "Frustration": [
        "frustrated", "irritated", "annoyed", "agitated", "exasperated", 
        "vexed", "discontented", "stuck", "fed up", "tired of", 
        "put out", "displeased", "exhausted", "unsettled", "disappointed", 
        "perturbed", "riled up", "bothered", "disheartened", "disgruntled", 
        "feeling trapped", "not getting anywhere", "hitting a wall", 
        "exhaustion", "cognitive overload", "feelings of helplessness", 
        "fighting against the current", "annoying tasks", 
        "feeling blocked", "lack of control", "challenging situations",
        "futile efforts", "feeling powerless", "irritation", 
        "persistent annoyances", "bottled up emotions", 
        "feeling cornered", "futile attempts"
    ],
    "Loneliness": [
        "lonely", "isolated", "alone", "friendless", "detached", 
        "forlorn", "desolate", "abandoned", "left out", "missing connection", 
        "disconnected", "withdrawn", "socially awkward", "feeling blue", 
        "lack of companionship", "longing for company", "yearning for friendship", 
        "nobody understands", "feeling invisible", "unseen", 
        "isolation", "feeling unheard", "solitary", 
        "need for connection", "heartache", "emptiness", 
        "nobody cares", "lost in a crowd", "feeling empty inside", 
        "unfulfilled relationships", "social fatigue"
    ],
    "Hopefulness": [
        "hopeful", "optimistic", "positive", "expectant", "encouraged", 
        "aspirational", "faithful", "looking forward", "bright future", 
        "light at the end of the tunnel", "motivated", "inspired", 
        "believing", "dreaming", "looking up", "full of life", 
        "good vibes", "feeling uplifted", "seeing possibilities", 
        "open to change", "faith in tomorrow", "anticipating joy", 
        "finding strength", "embracing new beginnings", 
        "looking for solutions", "persistence", "faith in self"
    ],
    "Happiness": [
        "happy", "joyful", "cheerful", "content", "delighted", "elated", 
        "gleeful", "feeling good", "smiling", "positive vibes", 
        "ecstatic", "blissful", "radiant", "thrilled", "grinning", 
        "chipper", "upbeat", "satisfied", "carefree", "optimistic", 
        "light-hearted", "thriving", "joyous", "bubbly", "full of joy", 
        "celebrating life", "pure happiness", "feeling blessed", 
        "good times", "laughter", "warmth", "playful", "grateful", 
        "living in the moment", "embracing happiness", 
        "making memories", "finding joy in little things"
    ],
    "Grief": [
        "grief", "mourning", "loss", "heartbroken", "sadness", 
        "suffering", "pain", "longing", "yearning", "remorse", 
        "bitterness", "despair", "disappointment", "regret", 
        "feeling empty", "unresolved feelings", "grieving", 
        "emotional pain", "tears", "heartache", "troubled", 
        "loss of connection", "nostalgia", "remembrance", 
        "finding closure", "difficult memories", 
        "finding solace", "searching for peace", "feeling incomplete", 
        "life after loss", "learning to cope", "processing emotions"
    ],
    "Confusion": [
        "confused", "uncertain", "perplexed", "bewildered", "mixed signals", 
        "lost", "disoriented", "puzzled", "unsettled", "muddled", 
        "unsure", "discombobulated", "caught off guard", 
        "unclear", "mind fog", "trapped in indecision", "cluttered mind", 
        "feeling stuck", "questioning everything", 
        "thinking in circles", "mental chaos", "grappling with thoughts",
        "overloaded with information", "lack of clarity", 
        "conflicting feelings", "feeling torn", "struggling to decide"
    ],
    
    # Psychopathic feelings and mental conditions
    "Narcissism": [
        "narcissistic", "self-centered", "grandiose", "entitled", 
        "self-important", "superior", "overly proud", 
        "self-absorbed", "thinking too highly of self", "self-serving", 
        "need for admiration", "lack of regard for others", "self-promotion", 
        "attention-seeking", "comparing to others", "excessive pride",
        "ego-driven", "delusions of grandeur", "need for control", 
        "grandiosity", "hyper-competitiveness", "sensitive to criticism",
        "manipulative tendencies", "using others for gain", 
        "superficial charm", "lack of empathy", "resentment towards others' success"
    ],
    "Lack of Empathy": [
        "lack of empathy", "unemotional", "cold-hearted", 
        "indifferent", "apathetic", "uncaring", 
        "unable to connect with others", "emotionally detached", 
        "disregard for feelings", "insensitive", "disconnected", 
        "not understanding", "emotionally shallow", "self-involved",
        "self-serving bias", "failure to understand consequences", 
        "emotional blindness", "limited emotional insight", 
        "unresponsive", "lack of compassion", "dismissive of others' feelings",
        "unwillingness to compromise", "self-absorbed behaviors"
    ],
    "Impulsivity": [
        "impulsive", "rash", "reckless", "spontaneous", 
        "hasty", "quick decisions", "flying off the handle", 
        "lack of foresight", "acting without thinking", "compulsive behavior",
        "urgency", "difficulty waiting", "inability to delay gratification", 
        "instant gratification", "excessive risk-taking", 
        "difficulty controlling impulses", "emotion-driven actions", 
        "blurt out", "erratic behavior", "lack of planning", 
        "need for immediate reward", "feeling out of control"
    ],
    "Manipulation": [
        "manipulative", "deceptive", "calculating", 
        "scheming", "coercive", "conniving", 
        "using others", "controlling", "playing mind games", 
        "exploiting vulnerabilities", "twisting the truth",
        "gaslighting", "emotional blackmail", "emotional manipulation", 
        "victim playing", "using guilt", "feigned ignorance", 
        "covert aggression", "misleading", "strategic deceit",
        "deliberate misunderstandings", "emotional exploitation"
    ],
    "Antisocial Behavior": [
        "antisocial", "disregard for others", "lawless", 
        "violent", "disruptive", "socially unacceptable", 
        "rebellious", "hostile to society", "nonconformist", 
        "criminal behavior", "lack of remorse", "manipulative tendencies",
        "troublesome behavior", "rejection of authority", 
        "violating social norms", "aggressive", "threatening", 
        "substance abuse", "disrespectful", "irresponsible", 
        "alienation from society", "social dysfunction"
    ],
    "Psychopathy": [
        "psychopathic", "sociopathic", "emotionally detached", 
        "remorseless", "unfeeling", "lack of conscience", 
        "manipulative tendencies", "no guilt", "thrill-seeking behavior", 
        "lack of long-term goals", "pervasive lying", "callousness", 
        "self-destructive behaviors", "difficulties in relationships", 
        "inability to form genuine connections", "superficial charm", 
        "shallow emotions", "irresponsibility", "exploitation of others",
        "need for stimulation", "failure to learn from experience"
    ],
    
    # Self-harm, suicidal thoughts, and self-obsession
    "Self-Harm": [
        "self-harm", "cutting", "burning", "self-injury", 
        "hurt myself", "self-destructive", "pain as relief", 
        "in need of release", "inflicting pain", "self-punishment", 
        "finding comfort in pain", "destructive behavior", "seeking pain",
        "risk-taking behavior", "cry for help", "feeling numb", 
        "emotional release", "hurt to feel alive", "self-sabotage",
        "dealing with emotional pain", "finding a way to cope",
        "misguided coping mechanisms", "overwhelmed by emotions"
    ],
    "Suicidal Thoughts": [
        "suicidal", "end it all", "wish I were dead", 
        "take my life", "kill myself", "suicide", 
        "hopelessness", "despair", "feeling trapped", 
        "wanting to escape", "life isn't worth living", "dark thoughts", 
        "feeling like a burden", "no way out", "thoughts of self-harm", 
        "wanting to disappear", "facing the end", "seeing no future",
        "reaching out for help", "struggling with despair", "no more pain",
        "unbearable suffering", "mental anguish", "wanting peace",
        "seeking relief", "drowning in sadness", "last resort", 
        "overwhelmed by thoughts of death"
    ],
    "Self-Obsession": [
        "self-obsessed", "self-absorbed", "selfish", 
        "self-centered", "narcissistic", "self-focused", 
        "constantly thinking about self", "looking inwards", 
        "self-promotion", "self-admiration", "self-aggrandizing", 
        "overly introspective", "self-fixation", "overthinking oneself", 
        "excessive self-analysis", "feeling superior", 
        "entitlement", "looking for validation", 
        "using others for self-gain", "need for constant reassurance"
    ],
    "Hopelessness": [
        "hopeless", "despairing", "lost hope", 
        "no way out", "pessimism", "feeling trapped", 
        "powerlessness", "inability to change", "loss of faith", 
        "giving up", "overwhelming darkness", "stagnation", 
        "stuck in a rut", "feeling lifeless", "endless struggle", 
        "feeling paralyzed", "constant disappointment", 
        "lack of progress", "resigned to fate", "burdened by reality"
    ],
    "Desperation": [
        "desperate", "panicked", "at the end of my rope", 
        "futile", "lost", "stuck", "feeling helpless", 
        "no options left", "wishing for a way out", 
        "overwhelmed by circumstances", "pleading", 
        "seeking a miracle", "hitting rock bottom", 
        "feeling defeated", "yearning for change", 
        "losing all hope", "on the verge of collapse"
    ],
    "Homicidal Ideation": [
        "homicidal", "kill", "murderous", "harm others", 
        "violent thoughts", "wanting to hurt", "thoughts of violence", 
        "desiring to end life", "intrusive thoughts about killing", 
        "fantasizing about death", "feelings of rage", 
        "demanding justice", "wanting revenge", "need for control",
        "aggressive impulses", "impulsive rage", "justified violence", 
        "destructive urges", "uncontrollable anger", "escape through harm"
    ],

    # Mixed feelings, emotional complexity, and dumb thoughts
    "Mixed Feelings": [
        "conflicted", "ambivalent", "torn", "both sides", 
        "bittersweet", "overwhelming emotions", "complex feelings", 
        "confusion about feelings", "mixed signals", "unclear", 
        "feeling both ways", "emotionally complex", "overwhelmed by options", 
        "struggling to decide", "feeling caught", "inconsistent emotions", 
        "complicated emotions", "not sure how to feel", "dilemma", 
        "difficult choices", "weighing options"
    ],
    "Dumb Thoughts": [
        "crazy ideas", "silly thoughts", "absurd notions", 
        "ridiculous thoughts", "thoughts that make no sense", 
        "random musings", "bizarre ideas", "foolish thoughts", 
        "light-hearted musings", "strange perceptions", "dizzying ideas", 
        "uncommon thoughts", "unfounded worries", "irrational beliefs", 
        "spontaneous thoughts", "mind wandering", "lack of focus", 
        "idle thinking", "seeking distractions", "lost in thought"
    ],

    # Everyday values in simple English
    "Values": [
        "love", "kindness", "respect", "honesty", "trust", 
        "friendship", "loyalty", "patience", "gratitude", 
        "understanding", "compassion", "generosity", "forgiveness", 
        "fairness", "integrity", "courage", "responsibility", 
        "hard work", "humility", "perseverance", "community", 
        "unity", "empathy", "positivity", "self-respect", 
        "creativity", "self-improvement", "growth", "caring", 
        "supportiveness", "inclusiveness", "tolerance", 
        "joyfulness", "happiness", "serenity", "mindfulness", 
        "flexibility", "acceptance", "resourcefulness", 
        "dedication", "open-mindedness", "authenticity", 
        "simplicity", "well-being", "collaboration", 
        "peacefulness", "humor", "passion", "playfulness", 
        "adventure", "self-awareness", "reliability", "sincerity", 
        "boundaries", "balance", "self-care", "wholeness", 
        "self-acceptance", "adaptability", "learning", "exploration", 
        "sustainability", "harmony", "authentic relationships", 
        "social responsibility", "safety", "support", "family", 
        "tradition", "fun", "discovery", "self-discipline", 
        "dedication", "authenticity", "personal growth", 
        "well-roundedness", "vision", "inspiration", 
        "meaningfulness", "balance", "positive reinforcement"
    ],
    # Expanded mental health concern categories with additional keywords
# concern_categories = {
    # Love and Attraction
    "Love": [
        "love", "affection", "romance", "passion", "adore", 
        "infatuation", "devotion", "attachment", "longing", 
        "heartfelt", "desire", "caring", "intimacy", 
        "relationship", "fondness", "sweetheart", "heartwarming", 
        "crush", "cherish", "emotion", "enamored", 
        "yearning", "being in love", "deep connection", 
        "spark", "chemistry", "affectionate", "dating", 
        "soulmate", "companion", "unconditional love"
    ],
    "Attraction": [
        "attraction", "drawn to", "chemistry", "fascination", 
        "allure", "magnetism", "enthralling", "captivated", 
        "mesmerized", "charisma", "irresistible", "appealing", 
        "infatuated", "magnetic pull", "intense interest", 
        "curiosity", "spark of interest", "admiration", 
        "crush", "romantic feelings", "compelling connection"
    ],
    
    # Dilemmas and Confusion
    "Dilemma": [
        "dilemma", "difficult choice", "hard decision", "tough call", 
        "conflicting priorities", "weighing options", "uncertainty", 
        "struggling to choose", "crossroads", "lost in choices", 
        "making sacrifices", "finding balance", "conflicted", 
        "ambiguity", "ethical dilemma", "heart vs. mind", 
        "wishy-washy", "feeling torn", "mixed feelings", 
        "overthinking decisions", "needing clarity", "split between options"
    ],
    "Confused Decisions": [
        "confused", "uncertain", "puzzled", "bewildered", 
        "unsure", "lost", "disoriented", "feeling stuck", 
        "questioning choices", "lack of clarity", "second-guessing", 
        "paralyzed by options", "decision fatigue", "mind clutter", 
        "indecisive", "trapped in thought", "grappling with choices", 
        "weighing pros and cons", "lost in thought", "doubtful"
    ],

    # Feelings related to Kill and Violence
    "Homicidal Thoughts": [
        "kill", "murder", "homicidal", "violence", 
        "violent thoughts", "harm others", "thoughts of rage", 
        "anger towards others", "desiring harm", "dark thoughts", 
        "intrusive thoughts about killing", "fantasizing about death", 
        "violent impulses", "aggressive thoughts", "outbursts", 
        "destructive tendencies", "retaliation", "revenge fantasies", 
        "loss of control", "wanting to lash out", "thoughts of destruction"
    ],
    
    # Mixed feelings and Emotional Complexity
    "Mixed Feelings": [
        "conflicted", "ambivalent", "torn", "bittersweet", 
        "overwhelming emotions", "confusion about feelings", 
        "unclear emotions", "feeling both ways", "emotional conflict", 
        "emotional complexity", "dueling feelings", "cognitive dissonance", 
        "overwhelmed by emotions", "struggling with feelings", 
        "complex feelings", "emotionally complicated", 
        "contradictory feelings", "sense of duality", 
        "inner turmoil", "difficult choices"
    ],
}

# Intensity modifiers
intensity_modifiers = {
    'extreme': 3, 'very': 2, 'really': 2, 'severely': 3, 'completely': 2,
    'totally': 2, 'always': 2, 'never': 2, 'constantly': 2, 'extremely': 3
}

# Severity words
severity_words = {
    'suicidal': 10, 'kill': 9, 'die': 8, 'harm': 8, 'hurt': 7, 'hopeless': 7,
    'desperate': 7, 'severe': 6, 'terrible': 6, 'horrible': 6, 'anxious': 5,
    'depressed': 6, 'scared': 5, 'afraid': 5, 'worried': 4, 'sad': 4,
    'upset': 4, 'stress': 4, 'tired': 3, 'uncomfortable': 3, 'uneasy': 3
}

def detect_polarity(text):
    result = sentiment_pipeline(text)[0]
    return result['label']

def extract_keywords(text):
    doc = nlp(text)
    keywords = {'entities': [], 'emotions': [], 'symptoms': [], 'actions': []}
    
    # Extract named entities
    for ent in doc.ents:
        keywords['entities'].append(ent.text)
    
    # Pattern matching lists
    emotion_patterns = ['feel', 'feeling', 'felt', 'anxiety', 'depression', 'stress', 'happy', 'sad', 'angry']
    symptom_patterns = ['cant sleep', "can't sleep", 'tired', 'exhausted', 'pain', 'ache', 'worried']
    action_patterns = ['kill', 'hurt', 'harm', 'help', 'need', 'want']
    text_lower = text.lower()
    
    # Process each token
    for token in doc:
        token_text = token.text.lower()
        if token_text in emotion_patterns:
            keywords['emotions'].append(token.text)
        if token_text in symptom_patterns:
            keywords['symptoms'].append(token.text)
        if token_text in action_patterns:
            keywords['actions'].append(token.text)
    
    # Check for multi-word patterns
    for pattern in symptom_patterns:
        if ' ' in pattern and pattern in text_lower:
            keywords['symptoms'].append(pattern)
    
    return keywords

def calculate_intensity(text, keywords):
    base_score = 1
    text_lower = text.lower()
    doc = nlp(text_lower)
    
    # Check for severity words
    max_severity = 0
    for word, score in severity_words.items():
        if word in text_lower:
            max_severity = max(max_severity, score)
    
    if max_severity > 0:
        base_score = max_severity
    
    # Apply intensity modifiers
    modifier_bonus = 0
    for modifier, value in intensity_modifiers.items():
        if modifier in text_lower:
            modifier_bonus += value
    
    # Consider number of symptoms and concerns
    symptom_count = len(keywords['symptoms'])
    emotion_count = len(keywords['emotions'])
    action_count = len(keywords['actions'])
    concern_bonus = min(2, (symptom_count + emotion_count + action_count) / 3)
    
    # Consider sentiment
    sentiment = detect_polarity(text)
    sentiment_modifier = 1.2 if sentiment == "NEGATIVE" else 0.8
    
    # Check for repetition
    word_counts = Counter([token.text.lower() for token in doc if token.text.lower() in severity_words])
    repetition_bonus = sum(0.5 for count in word_counts.values() if count > 1)
    
    # Calculate final score
    final_score = (base_score + modifier_bonus + concern_bonus + repetition_bonus) * sentiment_modifier
    final_score = max(1, min(10, final_score))
    
    return {
        "base_severity": base_score,
        "modifiers": modifier_bonus,
        "concern_count": concern_bonus,
        "sentiment_impact": sentiment_modifier,
        "repetition_impact": repetition_bonus,
        "final_score": round(final_score, 1)
    }

def classify_concern(text):
    concerns = []
    text_lower = text.lower()
    
    for category, keywords in concern_categories.items():
        if any(keyword in text_lower for keyword in keywords):
            concerns.append(category)
    
    if not concerns:
        concerns.append("General Mental Health")
    
    return concerns

def assess_risk(text, keywords):
    risk_level = "LOW"
    risk_factors = []
    
    high_risk_words = ['kill', 'death', 'suicide', 'hurt', 'harm']
    
    if any(word in text.lower() for word in high_risk_words):
        risk_level = "HIGH"
        risk_factors.append("High-risk words detected")
    
    if keywords['actions']:
        if any(action.lower() in high_risk_words for action in keywords['actions']):
            risk_level = "HIGH"
            risk_factors.append("Concerning actions detected")
    
    return {
        "level": risk_level,
        "factors": risk_factors
    }

def analyze_mental_health(text):
    keywords = extract_keywords(text)
    intensity = calculate_intensity(text, keywords)
    polarity = detect_polarity(text)
    concerns = classify_concern(text)
    risk_assessment = assess_risk(text, keywords)

    return {
        "input_text": text,
        "polarity": polarity,
        "detected_keywords": keywords,
        "identified_concerns": concerns,
        "intensity_analysis": intensity,
        "risk_assessment": risk_assessment,
        "timestamp": datetime.utcnow()
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('text', '')

        # Perform comprehensive analysis
        analysis_result = analyze_mental_health(text)
        
        # Insert into MongoDB
        analysis_collection.insert_one(analysis_result)

        # Return the analysis result as JSON
        return jsonify(analysis_result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_connection', methods=['GET'])
def test_connection():
    try:
        # Try a simple query
        test_data = analysis_collection.find_one()
        return jsonify(test_data if test_data else {"message": "No data found"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
