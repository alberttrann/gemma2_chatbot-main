#streamlit là framework tạo web app bằng python
#import os để cung cấp tools tương tác với hệ điều hành, chẳng hạn như quản lý tệp và thư mục
#from io import StringIO để thao tác với các luồng văn bản 
#import requests để cho phép HTTP requests để tương tác với APi và các máy chủ ngoài 
import streamlit as st
import os
from io import StringIO
import requests

#Streamlit setup
st.set_page_config(page_title='Gemma2 Chatbot', 
                    page_icon = "images/gemma_avatar.jpg",
                    initial_sidebar_state = 'auto')

#Background color
background_color = "#252740"
#Tạo ảnh avatar
avatars = {
    "assistant" : "images/gemma_avatar.jpg",
    "user": "images/user_avatar.png"
}

#Hiển thị chatbot header 
st.markdown("<h2 style='text-align: center; color: #3184a0;'>Gemma2 Chatbot</h2>", unsafe_allow_html=True)

#Hiển thị avatar bên sidebar 
with st.sidebar:
    st.image("images/gemma.jpg")

#Session state cho lịch sử chat - nếu không có tin nhắn trước đó được lưu thì khởi tạo session state bằng welcome message 
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

#Hiển thị chat messages 
for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                         avatar=avatars[message["role"]]):
        st.write(message["content"])

#Tạo function để clear chat history 
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]    
st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

#run_query function-Gửi POST request với user input đến backend API (FastAPI server ở http://127.0.0.1:8000/chat_messages)
#Nhận phản hồi từ backend và trả về
def run_query(input_text):
    """
    """
    data={'input_text': input_text}
    r = requests.post('http://127.0.0.1:8000/chat_messages', json = data)

#Check HTTP response status, nếu status code là 200(ok đó) thì content sẽ được parse dưới dạng json và phản hồi của chatbot("agent") sẽ được trích xuất từ phản hồi của API    
    if r.status_code == 200:
        print(r.content)
        agent_message = r.json()["agent"]

        return agent_message
    
    return "Error"

output = st.empty()
#Capture user input, thêm vào session state và hiển thị trong chat
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatars["user"]):
        st.write(prompt)

#Nếu tin nhắn cuối cùng là từ người dùng, nó sẽ gửi dữ liệu đầu vào đến backend (thông qua run_query) và hiển thị phản hồi của assistant.
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar=avatars["assistant"]):
        with st.spinner("Thinking..."):
            

            response = run_query(prompt)
#Render phản hồi của assistant real time
#một placeholder được tạo ra để hiển thị phản hồi 
#Vòng lặp mô phỏng quá trình hiển thị gia tăng của phản hồi(tạo hiệu ứng typing)
#Phản hồi cuối cùng sẽ được ghi vào trong placeholder dưới dạng Markdown
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response, unsafe_allow_html=True)
            placeholder.markdown(response, unsafe_allow_html=True)
#Update chat session state
    message = {"role": "assistant", 
               "content": response,
               "avatar": avatars["assistant"]}
    st.session_state.messages.append(message)