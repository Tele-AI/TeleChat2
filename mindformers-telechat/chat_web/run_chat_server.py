# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Run Chat Server

Usage:
    Start server:
        `python run_chat_server.py &> server.log &`
    End server:
        `kill -9 $(ps aux | grep "python run_chat_server.py" | grep -v grep | awk '{print $2}')`
"""
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException

from config.server_config import default_config
from predict_process import MindFormersInfer
from item import ChatCompletionResponse, ChatCompletionRequest, \
    ChatCompletionResponseChoice, ChatMessage, ChatCompletionResponseStreamChoice, \
    DeltaMessage, ChatErrorOutResponseStreamChoice, ChatErrorOutResponse, ChatErrorOutResponseChoice
from sse_starlette.sse import EventSourceResponse

from mindformers import logger

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None, swagger_ui_oauth2_redirect_url=None)
sem = None
ms_infer = None


@app.on_event('startup')
async def create():
    """Create semaphore and inference object on start up"""
    global ms_infer, sem
    sem = asyncio.Semaphore(1)
    ms_infer = MindFormersInfer()


# 健康探针
@app.get('/health')
async def health():
    """Check health of the server"""
    return {
        'code': 200
    }


@app.post("/generate")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion"""
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content
    logger.info('receive a request：{}'.format(request.messages))

    await sem.acquire()
    try:
        ms_infer.infer(query,
                       do_sample=request.do_sample,
                       temperature=request.temperature,
                       repetition_penalty=request.repetition_penalty,
                       top_k=request.top_k,
                       top_p=request.top_p,
                       max_length=request.max_length,
                       stream=request.stream)
        if request.stream:
            generate = _streaming_predict()
            return EventSourceResponse(generate, media_type='text/event-stream')

        response = ms_infer.get_res()
        logger.info(f"Responding to requests, generate result is {response}")
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=response),
            finish_reason='stop'
        )
        sem.release()
        return ChatCompletionResponse(choices=[choice_data], object="chat.completion")
    except ValueError as err:
        if request.stream:
            return EventSourceResponse(_error_generator(repr(err)), media_type='text/event-stream')
        choice_data = ChatErrorOutResponseChoice(
            index=0,
            message=repr(err),
            finish_reason="error"
        )
        sem.release()
        return ChatErrorOutResponse(choices=[choice_data], object="chat.completion")


def _streaming_predict():
    """Streaming predict"""
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    for new_response in ms_infer.get_res_iter():
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=new_response),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(choices=[choice_data], object="chat.completion.chunk")
    sem.release()
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))


def _error_generator(msg):
    """Error generator"""
    choice_data = ChatErrorOutResponseStreamChoice(
        index=0,
        message=msg,
        finish_reason="error"
    )
    chunk = ChatErrorOutResponse(choices=[choice_data], object="chat.completion.chunk")
    sem.release()
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))


if __name__ == "__main__":
    try:
        uvicorn.run(
            'run_chat_server:app',
            host=default_config['server']['host'],
            port=default_config['server']['port'],
            access_log=default_config['server']['access_log'],
            log_level=default_config['server']['uvicorn_level']
        )
    except OSError as e:
        logger.error(repr(e))
        logger.info("Stopping the server...")
