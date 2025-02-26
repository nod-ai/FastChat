import argparse
from collections import defaultdict
import datetime
import json
import os
import time
import uuid

import gradio as gr
import requests

from fastchat.conversation import get_default_conv_template, SeparatorStyle
from fastchat.constants import LOGDIR
from fastchat.utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
)
from fastchat.serve.gradio_patch import Chatbot as grChatbot
from fastchat.serve.gradio_web_server import (
    http_bot,
    set_global_vars,
    get_window_url_params,
    get_conv_log_filename,
    block_css,
    build_single_model_ui,
    no_change_btn,
    enable_btn,
    disable_btn,
    get_model_list,
    load_demo_single,
)
from fastchat.serve.inference import compute_skip_echo_len


logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_models = 2


def load_demo_side_by_side(url_params):
    states = (None,) * num_models
    dropdown_updates = (gr.Dropdown.update(visible=True),) * num_models

    return (
        states
        + dropdown_updates
        + (gr.Chatbot.update(visible=True),) * num_models
        + (
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Row.update(visible=True),
            gr.Row.update(visible=True),
            gr.Accordion.update(visible=True),
        )
    )


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    selected = 0
    if "compare" in url_params:
        selected = 1
    single_updates = load_demo_single(url_params)
    side_by_side_updates = load_demo_side_by_side(url_params)
    return (gr.Tabs.update(selected=selected),) + single_updates + side_by_side_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote. ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 3


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote. ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 3


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote. ip: {request.client.host}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 3


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    states = [state0, state1]
    for i in range(num_models):
        states[i].messages[-1][-1] = None
        states[i].skip_next = False
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    return [None] * num_models + [None] * num_models + [""] + [disable_btn] * 5


def add_text(state0, state1, text, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    states = [state0, state1]

    for i in range(num_models):
        if states[i] is None:
            states[i] = get_default_conv_template("vicuna").copy()

    if len(text) <= 0:
        for i in range(num_models):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 5
        )

    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            logger.info(f"violate moderation. ip: {request.client.host}. text: {text}")
            for i in range(num_models):
                states[i].skip_next = True
            return (
                states
                + [x.to_gradio_chatbot() for x in states]
                + [moderation_msg]
                + [
                    no_change_btn,
                ]
                * 5
            )

    text = text[:1536]  # Hard cut-off
    for i in range(num_models):
        states[i].append_message(states[i].roles[0], text)
        states[i].append_message(states[i].roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + [""]
        + [
            disable_btn,
        ]
        * 5
    )


def http_bot_all(
    state0,
    state1,
    model_selector0,
    model_selector1,
    temperature,
    max_new_tokens,
    request: gr.Request,
):
    logger.info(f"http_bot_all. ip: {request.client.host}")
    states = [state0, state1]
    model_selector = [model_selector0, model_selector1]
    gen = []
    for i in range(num_models):
        gen.append(
            http_bot(states[i], model_selector[i], temperature, max_new_tokens, request)
        )

    chatbots = [None] * num_models
    while True:
        stop = True
        for i in range(num_models):
            try:
                ret = next(gen[i])
                states[i], chatbots[i] = ret[0], ret[1]
                buttons = ret[2:]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + list(buttons)
        if stop:
            break


def build_side_by_side_ui():
    notice_markdown = """
# ⚔️  Compare Open Large Language Models Side-by-Side
Chat with two models side-by-side and vote for which one is better!

### Terms of use
By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
The demo works better on desktop devices with a wide screen.

### Choose two models to chat with
- [Vicuna](https://vicuna.lmsys.org): a chat assistant fine-tuned from LLaMA on user-shared conversations.
- [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/): a chatbot fine-tuned from LLaMA on user-shared conversations and open-source datasets.
- [OpenAssistant (oasst)](https://open-assistant.io/): a chat-based assistant for everyone.
- [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm): an instruction-tuned open LLM by Databricks.
- [ChatGLM](https://chatglm.cn/blog): an open bilingual dialogue language model | 开源双语对话语言模型
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html): a model fine-tuned from LLaMA on 52K instruction-following demonstrations.
- [LLaMA](https://arxiv.org/abs/2302.13971): open and efficient foundation language models.
"""

    learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""

    states = [gr.State() for _ in range(num_models)]
    model_selectors = [None] * num_models
    chatbots = [None] * num_models

    notice = gr.Markdown(notice_markdown)

    with gr.Row():
        for i in range(num_models):
            with gr.Column():
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=models[i] if len(models) > i else "",
                    interactive=True,
                    show_label=False,
                ).style(container=False)

    with gr.Row():
        for i in range(num_models):
            with gr.Column():
                chatbots[i] = grChatbot(elem_id="chatbot", visible=False).style(
                    height=550
                )

    with gr.Row(visible=False) as button_row:
        leftvote_btn = gr.Button(value="👈 Left is better", interactive=False)
        tie_btn = gr.Button(value="🤝 Tie", interactive=False)
        rightvote_btn = gr.Button(value="👉 Right is better", interactive=False)

    with gr.Row():
        with gr.Column(scale=20):
            textbox = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press ENTER",
                visible=False,
            ).style(container=False)
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button(value="Send", visible=False)

    with gr.Row() as button_row2:
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("Parameters", open=False, visible=True) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        max_output_tokens = gr.Slider(
            minimum=0,
            maximum=1024,
            value=512,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(learn_more_markdown)

    # Register listeners
    btn_list = [leftvote_btn, rightvote_btn, tie_btn, regenerate_btn, clear_btn]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, max_output_tokens],
        states + chatbots + btn_list,
    )
    clear_btn.click(clear_history, None, states + chatbots + [textbox] + btn_list)

    for i in range(num_models):
        model_selectors[i].change(
            clear_history, None, states + chatbots + [textbox] + btn_list
        )

    textbox.submit(
        add_text, states + [textbox], states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, max_output_tokens],
        states + chatbots + btn_list,
    )
    send_btn.click(
        add_text, states + [textbox], states + chatbots + [textbox] + btn_list
    ).then(
        http_bot_all,
        states + model_selectors + [temperature, max_output_tokens],
        states + chatbots + btn_list,
    )

    return (
        states,
        model_selectors,
        chatbots,
        textbox,
        send_btn,
        button_row,
        button_row2,
        parameter_row,
    )


def build_demo():
    with gr.Blocks(
        title="Chat with Open Large Language Models",
        theme=gr.themes.Base(),
        css=block_css,
    ) as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Single Model", id=0):
                (
                    a_state,
                    a_model_selector,
                    a_chatbot,
                    a_textbox,
                    a_send_btn,
                    a_button_row,
                    a_parameter_row,
                ) = build_single_model_ui()
                a_list = [
                    a_state,
                    a_model_selector,
                    a_chatbot,
                    a_textbox,
                    a_send_btn,
                    a_button_row,
                    a_parameter_row,
                ]

            with gr.Tab("Side-by-Side", id=1):
                (
                    b_states,
                    b_model_selectors,
                    b_chatbots,
                    b_textbox,
                    b_send_btn,
                    b_button_row,
                    b_button_row2,
                    b_parameter_row,
                ) = build_side_by_side_ui()
                b_list = (
                    b_states
                    + b_model_selectors
                    + b_chatbots
                    + [
                        b_textbox,
                        b_send_btn,
                        b_button_row,
                        b_button_row2,
                        b_parameter_row,
                    ]
                )

        url_params = gr.JSON(visible=False)

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [tabs] + a_list + b_list,
                _js=get_window_url_params,
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--moderate", action="store_true", help="Enable content moderation"
    )
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list(args.controller_url)
    set_global_vars(args.controller_url, args.moderate, models)

    logger.info(args)
    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )
