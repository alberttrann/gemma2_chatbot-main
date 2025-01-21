# Gemma2chatbot
Xây dựng một chatbot đơn giản bằng Streamlit và FastAPI với model Gemma 2 phiên bản 2 tỉ tham số đã đượce tune tối ưu cho mục đích trò chuyện. Setup này chưa đi kèm với các tối ưu về tốc độ ở phía frontend và backend(như asynchronous handling) và các tối ưu ở phía model như batching hay lượng tử hoá xuống FP16. Đây chỉ là minh hoạ một setup chatbot local đơn giản từ file model raw
streamlit run main.py để khởi chạy front-end 
uvicorn main:app --reload để khởi chạy back-end

Để pytorch compile được trên nhân cuda của GPU, cần phải cài CUDA Driver và cuDNN libraries. 
Trước khi cài, vào terminal gõ nvvc --version để check version CUDA
Nếu máy không có GPU rời mà chỉ chạy chatbot trên cpu thì bỏ qua bước này, lên thẳng pytorch để lấy cú pháp tải pytorch cho cpu

Về Python cần cài từ 3.11 đổ xuống để tương thích với Pytorch 
Nên mở VSCode bằng quyền admin để tạo môi trường ảo ít bị lỗi hơn. Tạo môi trường ảo  bằng python venv venv và kích hoạt môi trường ảo bằng \venv\Scripts\Activate

input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
Ở phần này trong model_interface.py, nếu là macbook chip M thì có thể đổi cuda thành mps, nếu Mac Intel hoặc máy không có card rtx thì để là cpu

Còn một cái nữa phải tải là visual build tools, chọn hết tất cả các mục  liên quan tới c++, và cũng cần tải Cargo - một Rust package manager để build wheel cho tokenizer, nếu không có thì lúc khởi chạy backend sẽ lỗi

pip install --requirements.txt để kéo hết các package theo đúng phiên bản tương thích
