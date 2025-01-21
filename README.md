# Gemma2chatbot
 Building a simple chatbot with Google's Gemma2-2b-it model, using Streamlit and FastAPI. This setup does not come with optimizations like batching or FP16 precision or asynchronous handling in the backend, just a plain, quickly-built-from-scratch chatbot from a raw model file 
streamlit run main.py để khởi chạy front-end 
uvicorn main:app --reload để khởi chạy back-end

Để pytorch compile được trên nhân cuda của GPU, cần phải cài CUDA Driver và cuDNN libraries. 
Trước khi cài vào terminal gõ nvidia --smi để check version CUDA
Nếu máy không có GPU rời mà chỉ chạy chatbot trên cpu thì bỏ qua bước này, lên thẳng pytorch để lấy cú pháp tải pytorch cho cpu

Về Python cần cài từ 3.11 đổ xuống để tương thích với Pytorch 
Nên mở VSCode bằng quyền admin để tạo môi trường ảo đỡ lỗi hơn, tạo bằng python venv venv và kích hoạt môi trường ảo bằng \venv\Scripts\Activate

input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
Ở phần này trong model_interface.py, nếu là macbook chip M thì có thể đổi cuda thành mps, nếu Mac Intel hoặc máy không có card rtx thì để là cpu

Còn một cái nữa phải tải là visual build tools, chọn hết mấy cái liên quan tới c++, với Cargo - một Rust package manager để build wheel cho tokenizer, nếu hong có là lúc khởi chạy backend nó sẽ lỗi 