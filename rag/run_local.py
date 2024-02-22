# 导入必要的库
from langchain_community.llms.tongyi import Tongyi
from interface import load_chain
import gradio as gr


class Model_center():
    """
    存储问答 Chain 的对象
    """

    def __init__(self, llm, vector_db_name="faiss", verbose=False):
        self.chain = load_chain(llm, vector_db_name=vector_db_name, verbose=verbose)

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            result = self.chain({"question": question})
            chat_history.append(
                (question, result["answer"]))
            return "", chat_history
        except Exception as e:
            return e, chat_history


def run_gradio(llm, vector_db_name="faiss", verbose=False):
    model_center = Model_center(llm, vector_db_name, verbose)

    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>InternLM</center></h1>
                    <center>你的专属助手</center>
                    """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=450, show_copy_button=True)
                # 创建一个文本框组件，用于输入 prompt。
                msg = gr.Textbox(label="Prompt/问题")

                with gr.Row():
                    # 创建提交按钮。
                    db_wo_his_btn = gr.Button("Chat")
                with gr.Row():
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot], value="Clear console")

            # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
            db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                msg, chatbot], outputs=[msg, chatbot])

        gr.Markdown("""提醒：<br>
        1. 初始化数据库时间可能较长，请耐心等待。
        2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """)
    gr.close_all()
    # 直接启动
    demo.launch()


if __name__ == "__main__":
    TONGYI_API_KEY = open("TONGYI_API_KEY.txt", "r").read().strip()
    verbose = True

    # 加载通义千问大语言模型
    llm = Tongyi(dashscope_api_key=TONGYI_API_KEY, temperature=0, model_name="qwen-turbo")
    run_gradio(llm, vector_db_name="faiss", verbose=verbose)
