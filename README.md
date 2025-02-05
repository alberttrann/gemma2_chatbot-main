## Hướng Dẫn Thiết Lập Chatbot Đơn Giản với Streamlit, FastAPI và Gemma 2B (Local)

Hướng dẫn này trình bày cách xây dựng một chatbot đơn giản sử dụng Streamlit (frontend) và FastAPI (backend), với mô hình Gemma 2 phiên bản 2 tỉ tham số đã được tinh chỉnh cho mục đích trò chuyện. Thiết lập này là một minh họa cơ bản về việc triển khai chatbot cục bộ từ file mô hình gốc, chưa bao gồm các tối ưu về tốc độ.

**Yêu Cầu Cài Đặt (Tùy Chọn cho GPU):**

Để PyTorch có thể biên dịch trên nhân CUDA của GPU, bạn cần cài đặt:

1.  **CUDA Driver:** Kiểm tra phiên bản CUDA hiện tại bằng lệnh `nvvc --version` trong terminal trước khi cài đặt.
2.  **cuDNN Libraries.**

**Lưu ý:** Nếu máy tính của bạn không có GPU rời, bạn có thể bỏ qua bước này và cài đặt phiên bản PyTorch cho CPU (tham khảo hướng dẫn trên trang chủ PyTorch để biết cú pháp cài đặt).

**Yêu Cầu về Python:**

*   Nên sử dụng Python phiên bản **3.11 trở xuống** để đảm bảo tương thích với PyTorch. Phiên bản khuyến nghị là 3.10.2

**Môi Trường Ảo (Khuyến Nghị):**

*   Nên mở VSCode với quyền admin để giảm thiểu lỗi khi tạo môi trường ảo.
*   Tạo môi trường ảo: `python -m venv venv`
*   Kích hoạt môi trường ảo: `.\venv\Scripts\Activate` (trên Windows) hoặc `source venv/bin/activate` (trên Linux/macOS)

**Điều Chỉnh Mã (cho Macbook Chip M):**

*   Trong file `model_interface.py`, dòng `input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")` có thể được thay đổi thành `input_ids = self.tokenizer(input_text, return_tensors="pt").to("mps")` nếu bạn đang sử dụng Macbook với chip M. Nếu không, hoặc nếu bạn dùng máy không có card RTX, hãy giữ nguyên là `"cpu"`.

**Công Cụ Bổ Sung (Quan Trọng):**

*   **Visual Studio Build Tools:** Cần tải và cài đặt, chọn **tất cả** các mục liên quan đến C++.
*   **Cargo (Rust Package Manager):** Cần thiết để build wheel cho tokenizer. Nếu thiếu, backend sẽ báo lỗi khi khởi động. [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)

**Cài Đặt Các Gói:**

*   Chạy lệnh `pip install -r requirements.txt` để cài đặt tất cả các gói cần thiết theo đúng phiên bản tương thích.

## **TẢI XUỐNG MÔ HÌNH**

Bạn có thể tải mô hình Gemma 2 từ hai nguồn sau:

1.  **Kaggle Models:** Truy cập [Gemma 2 Model Card](https://www.kaggle.com/models/google/gemma-2), chọn phiên bản `gemma-2-2b-it`, chọn "Transformer network" và tải xuống thư mục chứa model.
2.  **Download Script (Kagglehub):** Sử dụng script tải xuống được cung cấp khi chọn model từ model card:

    ```python
    import kagglehub

    # Tải phiên bản mới nhất
    path = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b-it")

    print("Đường dẫn đến file model:", path)
    ```

## **TRIỂN KHAI CLASS MODEL INTERFACE**

Class `ModelInterface` chịu trách nhiệm khởi tạo mô hình và tokenizer, đồng thời cung cấp chức năng tạo câu trả lời từ mô hình dựa trên đầu vào.

*   **Khởi tạo (`__init__`)**: Xác định đường dẫn đến model và thực hiện tải tokenizer và model ban đầu thông qua hàm `initialize_model`.  Tham số `max_new_tokens` được dùng để kiểm soát độ dài tối đa của câu trả lời trả về.  **Lưu ý:** Thay `"gemma-2-transformers-gemma-2-2b-it-v2"` bằng đường dẫn thực tế đến thư mục chứa model Gemma 2 đã tải về.

    ```python
    def __init__(self):
        self.path_to_model = "gemma-2-transformers-gemma-2-2b-it-v2"
        self.max_new_tokens = 128
        self.initialize_model()
    ```

*   **`initialize_model()`**: Tải tokenizer và model bằng các hàm `from_pretrained` từ class `AutoTokenizer` và `AutoModelForCausalLM` tương ứng.

    ```python
    def initialize_model(self):
        start_time = time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.path_to_model)
        tok_time = time()
        print(f"Tải tokenizer: {round(tok_time-start_time, 1)} sec.")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path_to_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        mod_time = time()
        print(f"Tải model: {round(mod_time-tok_time, 1)} sec.")
    ```

*   **`get_message_response()`**: Hàm này thực hiện việc truy vấn (infer) mô hình. Nó định nghĩa các `terminators` (điểm kết thúc), chuẩn bị đầu vào, chạy hàm `generate` của mô hình và trích xuất câu trả lời từ văn bản đầu ra. Hàm `clean_answer` sẽ loại bỏ đầu vào ban đầu và các token đặc biệt như `<|eot_id|>` khỏi câu trả lời cuối cùng. Kết quả trả về là một JSON chứa đầu vào, câu trả lời đã được làm sạch và thời gian thực thi.

    ```python
    def get_message_response(self, input_text):
        start_time = time()

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("mps")

        outputs = self.model.generate(
                **input_ids,
                do_sample=True,
                top_k=10,
                temperature=0.1,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=terminators[0]
                )
        end_time = time()
        answer = self.clean_answer(f"{self.tokenizer.decode(outputs[0])}",
                                   input_text)

        print(f"Tổng thời gian phản hồi: {round(end_time-start_time, 1)} sec.")

        return {
                "input": input_text,
                "response": answer,
                "response_time":  f"{round(end_time-start_time, 1)} sec."
        }
    ```

## **BACKEND ĐƠN GIẢN VỚI FASTAPI**

Backend được xây dựng với FastAPI. Đầu tiên, một instance của `ModelInterface` được tạo. Sau đó, một endpoint `/chat_messages/` được định nghĩa để nhận yêu cầu (request) từ giao diện người dùng (frontend) và trả về phản hồi từ mô hình.

```python
model_interface = ModelInterface()

@app.post("/chat_messages/")
def chat_messages(input: Input):
    agent_response = model_interface.get_message_response(input_text=input.input_text)
    print(agent_response)
    return {"agent": agent_response["response"]}
```

## **FRONTEND ĐƠN GIẢN VỚI STREAMLIT**

Frontend sử dụng Streamlit để tạo giao diện chatbot và thu thập đầu vào từ người dùng.

*   Hàm `run_query()` gửi yêu cầu (request) đến endpoint `/chat_messages` của backend và trả về tin nhắn từ agent (mô hình).

    ```python
    def run_query(input_text):
        """
        Gửi yêu cầu đến endpoint chat_messages và trả về tin nhắn từ agent.
        """
        data={'input_text': input_text}
        r = requests.post('http://127.0.0.1:8000/chat_messages', json = data)

        if r.status_code == 200:
            print(r.content)
            agent_message = r.json()["agent"]
            return agent_message

        return "Error"
    ```

*   Giao diện chatbot chính: Sử dụng `st.chat_input()` để thu thập tin nhắn từ người dùng, sau đó thêm tin nhắn vào chat pane và gọi hàm `run_query()` để lấy phản hồi từ mô hình.

    ```python
    output = st.empty()
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=avatars["user"]):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar=avatars["assistant"]):
            with st.spinner("Thinking..."):

                response = run_query(prompt)

                placeholder = st.empty()
                full_response = ""
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response, unsafe_allow_html=True)
                placeholder.markdown(response, unsafe_allow_html=True)

        message = {"role": "assistant",
                   "content": response,
                   "avatar": avatars["assistant"]}
        st.session_state.messages.append(message)
    ```

*   Streamlit sẽ rerun toàn bộ code mỗi khi có hành động từ người dùng. Chat session được lưu trữ trong `st.session_state.messages`. Đoạn code sau khởi tạo chat session và hiển thị các tin nhắn đã có.

    ```python
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I assist you today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"],
                             avatar=avatars[message["role"]]):
            st.write(message["content"])
    ```

## **KHỞI CHẠY ỨNG DỤNG**

Để khởi chạy ứng dụng:

*   **Backend:** `uvicorn main:app --reload`
*   **Frontend:** `streamlit run main.py`
```
