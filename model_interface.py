
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from time import time


#Tạo system message 
#Set path tới pre-trained model gemma 2 và khởi tạo bằng cách load tokenizer và model 
#Set token mà model có thể generate trong 1 lần, do context window của model tới tận 8192 nên mình set max lun
class ModelInterface(object):

    def __init__(self):
        self.system_message = """
            You are an AI agent tasked to answer general questions in 
            a simple and short way.    
    """
        self.path_to_model = "C:\\Users\\alberttran\\Downloads\\gemma-2-transformers-gemma-2-2b-it-v2"
        self.max_new_tokens = 512
        self.initialize_model()

#Load tokenizer và model, sử dụng thư viện transformers của huggingface 
#AutoTokenizer là hàm để tự lựa tokenizer phù hợp dựa vào model file trong path mình đặt vào để chuyển text thành numerical representations(input_ids)
#self.model trở xuống là phần load cái model, AutoModelForCausalLLM là class cho causal language modelling, dùng cho text generation
#return_dict là đảm bảo output đưỢc return như trong dictionary, low gì đó là để tối ưu memory lúc loading
#device map là để map model với gpu/cpu,trust là để load custom model architectures, còn dưới nữa là để log time mất để load cái model
    def initialize_model(self):
        start_time = time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_model)
        tok_time = time()
        print(f"Load tokenizer: {round(tok_time-start_time, 1)} sec.")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path_to_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        mod_time = time()
        print(f"Load model: {round(mod_time-tok_time, 1)} sec.")

#Clean up raw mode-generated output để cho nó user-friendly hơn 
#bỏ user input ra khỏi response nếu có -> strip bớt các token đánh dấu cuộc trò chuyện -> trim leading whitespace bằng lstrip -> trả về cleaned-up answer
    @staticmethod
    def clean_answer(answer, input_text):
        answer = answer.replace(input_text,"")
        answer = answer.replace("<end_of_turn>", "")
        answer = answer.replace("<bos>", "")
        answer = answer.replace("<eos>", "")
        answer = answer.lstrip()
        return answer

#Generate response
#1. Tokenize user input thành input_ids(pytorch tensors) và chuyển tensors sang gpu
#2. gọi method generate cho text generation
# do_sample là cho lấy mẫu ngẫu nhiên, topk temp top p chắc mọi người xem youtube nha
#max.. giới hạn response length, eos_token_id là để dừng generation khi đụng end-of-sequence token
#pad_token_id là để fill unused token slot bằng padding 
#cuối cùng là chuyển tokenized output của model thành dạng người đọc đƯợc 
    def get_message_response(self, input_text):
        start_time = time()

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
                **input_ids,
                do_sample=True,
                max_length=768,
                top_k=40,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=terminators[0]
                )
        end_time = time()
        answer = self.clean_answer(f"{self.tokenizer.decode(outputs[0])}",
                                   input_text)

        print(f"Total response time: {round(end_time-start_time, 1)} sec.")


        return {
                "input": input_text,
                "response": answer,
                "response_time":  f"{round(end_time-start_time, 1)} sec."
                }