import json
import nltk
import spacy
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def ner_score(entities, total_tokens):
    entity_tokens = sum(len(ent.split()) for ent in entities)
    print("entity tokens: count",entity_tokens)
    return entity_tokens / total_tokens if total_tokens > 0 else 0

def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [f"{ent.text} ({ent.label_})" for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE"]]

# def extract_entities(text):
#     model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForTokenClassification.from_pretrained(model_name)
#     ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
#     entities = ner_pipeline(text)
#     allowed_labels = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}
#     return [f"{ent['word']} ({allowed_labels.get(ent['entity_group'], ent['entity_group'])})"
#             for ent in entities if ent['entity_group'] in allowed_labels]

def extractive(text):
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences)

def abstractive(text, entities):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def choose_summarizer(text,json_file_path):
    tokens = text.split()
    entities = extract_entities(text)
    print("len: ", len(entities))
    score = ner_score(entities, len(tokens))
    print("ner score",score)
    entity_tokens = sum(len(ent.split()) for ent in entities)

    if entity_tokens >= 15:
        print(f"Choosing extractive")
        return extractive(text), 0, score
    else:
        with open(json_file_path, "r") as file:
            data = json.load(file)
            next_goal = data["history"][-1]["model_output"]["next_goal"]
            text = data["history"][-1]["model_output"]["action"][0]["done"]["text"]
            extracted_content = data["history"][-1]["result"][0]["extracted_content"]
            d = f"{next_goal} \n {text} \n {extracted_content}"
            new_entities = extract_entities(d)
        print("Choosing abstractive")

        return abstractive(d, new_entities), 1, score

def generate_summary_prompt(json_path: str):
    with open(json_path, "r") as file:
        json_data = json.load(file)
        # data["history"][-1]["result"][0]["extracted_content"]
    final_text_summary = " "
    action = " "
    if json_data["history"][-1]["result"][0]["extracted_content"]:
        final_text_summary = json_data["history"][-1]["result"][0]["extracted_content"]

    if json_data["history"][-1]["model_output"]["action"][0]["done"]["text"]:
        action = json_data["history"][-1]["model_output"]["action"][0]["done"]["text"]

    process_summary_text =  f"{action}"
    prompt = f"""{process_summary_text}\n{final_text_summary}"""
    return prompt.strip(), process_summary_text, final_text_summary

# if __name__ == "__main__":
#     json_file_path = "/home/zaid/Downloads/voice-agent/json_summarizer/summarizer/raman.json"
    # json_new = "/home/zaid/Downloads/voice-agent/raman-hospitals-history.json"
    # original_text,one,two = generate_summary_prompt(json_new)
    # # print(f"one: {one}")
    # # print(f"two: {two}")
    # text = '''I have successfully completed your request to find multi-specialty hospitals near Paschim Vihar, New Delhi. Using Google search and directory sites like Justdial, I extracted a list of over 10 relevant hospitals, filtered the top 5 based on the highest number of reviews for reliability, and verified online appointment options for each via their official websites. All top 5 hospitals offer online booking or consultation features. Here is a summary of the top 5:\n\n1. **Sir Ganga Ram Hospital** (Rating: 4.2/5, Reviews: 34,343, Address: Sir Ganga Ram Hospital Marg, Old Rajinder Nagar, New Delhi - Near Paschim Vihar), Specialties: 71+ including Cardiac, Oncology, etc. Online Appointments: Yes, via 'Book Appointment' button, callback form, and department-specific scheduling on sgrh.com.\n\n2. **BLK-MAX Super Speciality Hospital** (Rating: 4.4/5, Reviews: 30,000+, Address: Pusa Road, Rajinder Nagar, New Delhi - Close to Paschim Vihar), Specialties: Comprehensive multi-specialty including Cancer Care, Heart, etc. Online Appointments: Yes, direct booking form, OPD scheduling, and consultations on blkmaxhospital.com.\n\n3. **Indraprastha Apollo Hospital** (Rating: 4.3/5, Reviews: ~25,000, Address: Sarita Vihar, New Delhi - Accessible from Paschim Vihar), Specialties: Multi-specialty with 50+ departments like Neurology, Orthopedics. Online Appointments: Yes, book appointments, health checkups, second opinions via apollohospitals.com/delhi.\n\n4. **Indian Spinal Injuries Hospital - Venkateshwar** (Rating: 4.4/5, Reviews: 22,000+, Address: Sector 18A, Dwarka, New Delhi - Near Paschim Vihar), Specialties: Multi-specialty including Spine, Orthopedics, Critical Care. Online Appointments: Yes, OPD scheduler, treatment advice form, and consultations on venkateshwarhospitals.com.\n\n5. **Max Super Speciality Hospital** (Rating: 4.1/5, Reviews: 13,000+, Address: Shalimar Bagh, New Delhi - Very close to Paschim Vihar), Specialties: 28+ including Oncology, Robotic Surgery, Cardiac. Online Appointments: Yes, appointment booking, doctor search, WhatsApp scheduling via maxhealthcare.in.\n\nFor the full list of extracted hospitals (10+ with detailed addresses, ratings, and more), specialties, and complete verification notes, please refer to the attached results.md file. All hospitals are multi-specialty and within or near Paschim Vihar for accessibility.", 'success': True, 'files_to_display': ['results.md']}}]'''
    # ent = extract_entities(text)
    # abs = abstractive(text,ent)
    # ext = extractive(text)
    # print(abs)
    # print(ext)
    # one_ = extractive(one)    # summary, model_used, ner_ratio = choose_summarizer(original_text)
    # # print(f"{model_used} model used")
    # # print("NER Ratio:", ner_ratio)
    # # print("Final: \n", summary)
    # two = two.replace("\n", " ")
    # two_entities = extract_entities(two)
    # two_ = abstractive(two, two_entities)
    #
    # final = f"{one_} \n\n {two_}"
    # print(final)
    # # summary, model_used, ner_ratio = choose_summarizer(original_text)
    # # print(f"{model_used} model used")
    # # print("NER Ratio:", ner_ratio)
    # # print("Final: \n", summary)
