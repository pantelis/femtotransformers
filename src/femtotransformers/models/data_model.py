from pydantic import BaseModel

class EncodeInput(BaseModel): 
    text: str

class EncodeOutput(BaseModel):
    encoded_text: list

class DecodeInput(BaseModel):
    encoded_text: list

class DecodeOutput(BaseModel):
    decoded_text: str
