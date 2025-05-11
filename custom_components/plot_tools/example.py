import gradio as gr

def plot_tools():
    return gr.Markdown("Plot Tools Component")

demo = gr.Interface(fn=plot_tools, inputs=None, outputs="markdown")

if __name__ == "__main__":
    demo.launch()
