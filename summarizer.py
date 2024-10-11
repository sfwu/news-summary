from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_dir="./fine_tuned_model"

def load_fine_tuned_model():
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return model,tokenizer

def prepare_input_for_generation(article):
    return f"{article} </s>"

def extract_summary(result):
    return result.split('</s>')[-1].strip()

def combined_score(summary, score, length_weight=0.5):
    length = len(summary)
    return score / (length ** length_weight)

def summarize(model,
              tokenizer,
              article,
              max_length=250,
              top_k=5,
              top_p=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_text = prepare_input_for_generation(article)
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

    generated_sequences = model.generate(
        input_ids,
        max_length=max_length,  # Set maximum length for the generated summary
        top_k=top_k,              # Limit the number of highest probability tokens
        top_p=top_p,             # Use nucleus sampling
        num_return_sequences=top_k,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True
    )

    decoded_sequences = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated_sequences.sequences]
    generated_summaries = [extract_summary(result) for result in decoded_sequences]
    scores = generated_sequences.scores
    average_scores = [torch.mean(torch.softmax(score, dim=-1)).item() for score in scores]
    scored_summaries = [(summary, combined_score(summary, average_scores[i])) for i, summary in enumerate(generated_summaries)]
    best_summary = sorted(scored_summaries, key=lambda x: -x[1])[0][0]

    return best_summary


def main():
    input_article = "the bishop of the fargo catholic diocese in north dakota has exposed potentially hundreds of church members in fargo, grand forks and jamestown to the hepatitis a virus in late september and early october. the state health department has issued an advisory of exposure for anyone who attended five churches and took communion. bishop john folda (pictured) of the fargo catholic diocese in north dakota has exposed potentially hundreds of church members in fargo, grand forks and jamestown to the hepatitis a . state immunization program manager molly howell says the risk is low, but officials feel it is important to alert people to the possible exposure. the diocese announced on monday that bishop john folda is taking time off after being diagnosed with hepatitis a. the diocese says he contracted the infection through contaminated food while attending a conference for newly ordained bishops in italy last month. symptoms of hepatitis a include fever, tiredness, loss of appetite, nausea and abdominal discomfort. fargo catholic diocese in north dakota (pictured) is where the bishop is located ."
    print(summarize(input_article))

if __name__ == '__main__':
    main()
