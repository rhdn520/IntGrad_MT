import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class HFInstructModelInterpolation():
    def __init__(self, model_id="Qwen/Qwen2-72B-Instruct", quantization=True) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
        ]
        if quantization:
            self.quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=self.quant_config, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    def interpolate(self, sent1:str, sent2:str, max_retry=3):
        print(f'Interpolating with {self.model_id}...')  
        retry_count = 0

        messages = [
                {
                "role": "user",
                "content": "I will give you two sentences. Can you gradually change the first sentence to make it exactly the same as the second sentence? Just give me the sentences and don't provide additional comments.\nSentence1: Surprisingly often, animals show up uninvited at sporting events. Sometimes, it gets a little weird.\nSentence2: Michael D. Cohen tried to parry attacks on his honesty as Todd Blanche, Donald J. Trump’s lawyer, tried to destroy his credibility with jurors."
                },
                {
                "role": "assistant",
                "content": "Surprisingly often, animals show up uninvited at sporting events. Sometimes, it gets a little weird.\nOften, unexpected events occur during live events, and sometimes these can get quite weird.\nAt public events, such as sports or trials, unexpected guests often cause disturbances, leading to weird situations.\nIn public spectacles, like sports or courtrooms, unexpected participants can disrupt the normal proceedings in strange ways.\nDuring public hearings, like in court, surprising accusations and defenses can emerge, often causing odd disruptions.\nIn courtroom battles, unexpected arguments and witnesses can often turn the proceedings weird as each side tries to undermine the other.\nIn legal battles, lawyers frequently confront each other with surprising tactics to challenge credibility, which can make the proceedings seem strange.\nMichael D. Cohen, during his trial, encountered surprising tactics by Todd Blanche, Donald J. Trump’s lawyer, aiming to disrupt his credibility in unusual ways.\nMichael D. Cohen tried to parry attacks on his honesty as Todd Blanche, Donald J. Trump’s lawyer, tried to destroy his credibility with jurors."
                },
                {
                "role": "user",
                "content": "I will give you two sentences. Can you gradually change the first sentence to make it exactly the same as the second sentence? Just give me the sentences and don't provide additional comments.\nSentence1: Georgia’s Parliament overrode a presidential veto to give final approval on Tuesday evening to a contentious bill that has plunged the country into a political crisis and threatened to derail the pro-Western aspirations of many Georgians in favor of closer ties with Russia.\nSentence2: At that time, the buildings were abandoned and inaccessible to the public, and the famous ferry that shuttled immigrants from the island to Manhattan sank in a storm in 1968."
                },
                {
                "role": "assistant",
                "content": "Georgia’s Parliament overrode a presidential veto to give final approval on Tuesday evening to a contentious bill that has plunged the country into a political crisis and threatened to derail the pro-Western aspirations of many Georgians in favor of closer ties with Russia.\nThe decision made by Georgia’s Parliament on Tuesday evening, to override a presidential veto and approve a controversial bill, has sparked a political crisis in the country and raised concerns about the impact on the country's pro-Western stance and relations with Russia.\nThe controversial bill approved by Georgia’s Parliament on Tuesday, despite a presidential veto, has caused a political crisis and stirred debates about the country’s Western alignment and its relation with Russia.\nThe unilateral move by Georgia’s Parliament on Tuesday to push through a divisive bill despite opposition from the president has led to a political crisis and raised concerns about the nation's Western affiliations and ties with Russia.\nThe recent decision by Georgia’s Parliament to defy the presidential veto and pass a controversial bill has created a political turmoil, questioning the country’s Western orientation and relationship with Russia.\nGeorgia’s Parliament's decision to override the presidential veto and approve the contentious bill has caused a political crisis in the country, prompting discussions on the impact on its Western ties and relations with Russia.\nGeorgia’s Parliament’s act of rejecting the presidential veto and endorsing the controversial bill on Tuesday has resulted in a political crisis, with implications for the country's pro-Western stance and connections with Russia.\nGeorgia’s Parliament moved forward with finalizing a contentious bill on Tuesday, overriding the presidential veto, which has triggered a political crisis and raised concerns about the nation's Western aspirations and relationship with Russia.\nAt that time, the buildings were abandoned and inaccessible to the public, and the famous ferry that shuttled immigrants from the island to Manhattan sank in a storm in 1968."
                },
                {
                "role":"user",
                "content": "I will give you two sentences. Can you gradually change the first sentence to make it exactly the same as the second sentence? Just give me the sentences and don't provide additional comments.\nSentence1: The cake has become one of Jamaica’s most popular desserts, often found in various stores.\nSentence2: Some plans might cover only generic versions of certain contraceptives, but patients are still entitled to coverage of a specific product that their providers deem medically necessary."
                },
                {
                "role":"assistant",
                "content":"The cake has become one of Jamaica’s most popular desserts, often found in various stores.\nSome plans might only cover basic versions of popular desserts like the cake, while others are more comprehensive.\nIn many cases, plans might cover only generic versions of popular products, but specific varieties might still be available.\nSome plans might cover only generic versions of certain products, with exceptions allowed for specific needs.\nSome plans might cover only generic versions of certain healthcare products, but specific options are still accessible if necessary.\nSome plans might cover only generic versions of certain contraceptives, but patients can still get specific types if deemed necessary.\nSome plans might cover only generic versions of certain contraceptives, but patients are still entitled to coverage of a specific product that their providers deem medically necessary."
                },
                {
                "role": "user",
                "content": f"I will give you two sentences. Can you gradually change the first sentence to make it exactly the same as the second sentence? Just give me the sentences.\nSentence1: {sent1}\nSentence2: {sent2}"
                }
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        while(retry_count < max_retry):

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=self.terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

            response = outputs[0][input_ids.shape[-1]:]
            response_text = self.tokenizer.decode(response, skip_special_tokens=True)
            intp_sents = [s.strip() for s in response_text.split('\n')]
            if(intp_sents[0] == sent1 and intp_sents[-1] == sent2):
                return intp_sents
            else:
                intp_sents = ['ERROR']
                retry_count += 1 

        return intp_sents
    