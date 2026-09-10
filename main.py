from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import cv2
import numpy as np

from processor import remove_ruled_lines, add_letterhead


app = FastAPI(
    title="Image Processing API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Image Processing API is running"
    }


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # 1. Validate file
    # -----------------------------------------------------

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are allowed"
        )

    # -----------------------------------------------------
    # 2. Read uploaded image
    # -----------------------------------------------------

    contents = await file.read()

    # Convert bytes -> NumPy array
    np_array = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    # Convert NumPy array -> OpenCV image
    image = cv2.imdecode(
        np_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )

    # -----------------------------------------------------
    # 3. Process image
    # -----------------------------------------------------

    lines = remove_ruled_lines(
        image
    )

    
    result  = add_letterhead(
        lines
    )


    # -----------------------------------------------------
    # 4. Convert processed image -> PNG
    # -----------------------------------------------------

    success, encoded_image = cv2.imencode(
        ".png",
        result
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode processed image"
        )

    # -----------------------------------------------------
    # 5. Return image directly
    # -----------------------------------------------------

    return Response(
        content=encoded_image.tobytes(),
        media_type="image/png"
    )