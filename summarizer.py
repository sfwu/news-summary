from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_dir="./fine_tuned_model"

def load_fine_tuned_model():
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return model,tokenizer

def prepare_input_for_generation(article):
    return f"{article} </s>"

def extract_summary(result):
    return result.split('</s>')[-1].strip()

def summarize(model,
              tokenizer,
              article,
              max_length=100,
              top_k=20,
              top_p=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_text = prepare_input_for_generation(article)
    input_ids = tokenizer.encode(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)

    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        early_stopping=True
    )
    all_result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    generated_summary = extract_summary(all_result)
    return generated_summary


def main():
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    input_article = "the bishop of the fargo catholic diocese in north dakota has exposed potentially hundreds of church members in fargo, grand forks and jamestown to the hepatitis a virus in late september and early october. the state health department has issued an advisory of exposure for anyone who attended five churches and took communion. bishop john folda (pictured) of the fargo catholic diocese in north dakota has exposed potentially hundreds of church members in fargo, grand forks and jamestown to the hepatitis a . state immunization program manager molly howell says the risk is low, but officials feel it is important to alert people to the possible exposure. the diocese announced on monday that bishop john folda is taking time off after being diagnosed with hepatitis a. the diocese says he contracted the infection through contaminated food while attending a conference for newly ordained bishops in italy last month. symptoms of hepatitis a include fever, tiredness, loss of appetite, nausea and abdominal discomfort. fargo catholic diocese in north dakota (pictured) is where the bishop is located ."

    print(summarize(model,tokenizer,input_article))

if __name__ == '__main__':
    main()
