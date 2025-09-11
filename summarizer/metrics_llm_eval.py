from bert_score import score
import textstat
import nltk
from nltk.util import ngrams
import math
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from numba.cuda.printimpl import print_item
from typing_extensions import final
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# from workspace.bart_ai import summary_text

nltk.download('punkt')
def score_summary(original_text: str, generated_summary: str):
    # BERTScore
    P, R, F1 = score([generated_summary], [original_text], lang="en", verbose=False)
    bert_f1 = F1.item()

    # Flesch Reading Ease (normalized)
    flesch_score = textstat.flesch_reading_ease(generated_summary) / 100

    # Compression Ratio
    len_original = len(original_text.split())
    len_summary = len(generated_summary.split())
    compression_ratio = len_summary / len_original if len_original > 0 else 0

    # Coverage
    def coverage(src, summ):
        src_words = set(src.lower().split())
        summ_words = set(summ.lower().split())
        return len(src_words & summ_words) / len(src_words) if src_words else 0

    coverage_score = coverage(original_text, generated_summary)

    # METEOR
    ref_tokens = word_tokenize(original_text.lower())
    cand_tokens = word_tokenize(generated_summary.lower())
    meteor = meteor_score([ref_tokens], cand_tokens)

    # Final Weighted Score
    final = (
        0.5 * bert_f1 +
        0.15 * meteor +
        0.15 * coverage_score +
        0.1 * flesch_score +
        0.1 * compression_ratio
    )

    return {
        "bert_f1": bert_f1,
        "flesch_score": flesch_score,
        "compression_ratio": compression_ratio,
        "coverage": coverage_score,
        "meteor": meteor,
        "final_score": final
    }

# if __name__ == "__main__":
#     original_text = " Extracted 5-star luxury hotels in Dubai.. Summarize findings to identify the best hotel.. Based on Booking.com data, JW Marriott Hotel Marina is the top choice due to highest rating and prime location.. Actions: Features: Directly linked to Dubai Marina Mall, Near the beach, World-class service; Location: Dubai Marina, Sheikh Zayed Road; Name: JW Marriott Hotel Marina; Rating: 9.5 | Features: 1.2 miles of private beach, Luxury resort; Location: Al Sufouh; Name: Jumeirah Al Qasr Dubai; Rating: 9.4 | Features: Steps from Marina Beach, Modern luxury; Location: Jumeirah Beach Residence; Name: FIVE LUXE; Rating: 9.4 | Features: 1 km from downtown, City skyline views; Location: Business Bay; Name: The St. Regis Downtown Dubai; Rating: 9.2 | Features: Private sandy beach, Iconic luxury resort; Location: Palm Jumeirah; Name: Atlantis, The Palm; Rating: 9.2"
#
#     generated_summary1 = '''Find the best luxury hotel in Dubai with the highest customer rating and a prime
# location. The JW Marriott Hotel Marina is located in Dubai Marina, Sheikh Zayed
# Road. The Jumeirah Al Qasr Dubai is a modern luxury resort located in Al Sufouh.
# The St. Regis Downtown Dubai is an iconic luxury hotel located in Business Bay.
# '''
#     generated_summary2 = "Based on Booking.com data, JW Marriott Hotel Marina is the top choice due to highest rating and prime location. Extracted 5-star luxury hotels in Dubai. Summarize findings to identify the best hotel. Based on Bookings.comData, Jw Marriott HotelMarina is thetop choice due. to highest ratings and prime locations."
#     generated_summary1 = generated_summary1.replace("\n", " ")
#     metrics2 = score_summary(original_text, generated_summary1)
#     print(f"Final: {final_score}")
#
#
#     metrics = score_summary(original_text, generated_summary2)
#     print(f"Final: {metrics["final_score"]}")
