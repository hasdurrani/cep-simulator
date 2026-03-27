from pydantic import BaseModel


class CEPNode(BaseModel):
    cep_id: str
    cep_family: str
    cep_label: str
    cep_description: str
    active: bool = True
