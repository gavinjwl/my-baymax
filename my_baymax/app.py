from chainlit.utils import mount_chainlit
from fastapi import FastAPI

app = FastAPI()


@app.get("/app")
def read_main():
    '''
    In the example above, we have a FastAPI application with a single endpoint /app.
    '''
    return {"message": "Hello World from main app"}

# We mount the Chainlit application my_cl_app.py to the /chainlit path.
mount_chainlit(app=app, target="chainlit_app.py", path="/chainlit")
