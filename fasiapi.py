# Advantages of fast api
# 1. data valiadation
# 2. auto documentation
 
 from fastapi import FastAPI, Path


 app = FastAPI()

 '''
 ENDPOINT:
    An API end point is the point of entry in a communication
    channel when two systems are interacting. It referes to
    touchpoints of the communication between an API and a server
'''

# 4 core endpints
# GET: Return information
# POST: Sending information or creating something new
# PUT: Update something if something is existed
# DELETE: just delete

@app.get('/')
def home():
    return {"Data":"Test"}

@app.get('/about')
def about():
    return {"Data": "About"}

inventory = {
    1: {
        'name': 'milk',
        'price': 3.99,
        'brand': 'regular'
    }
}

# @app.get('/get-item/{item_id}/{name}')
# def get_item(item_id: int, name: str = None):
#     return inventory[item_id]

# path
@app.get('/get-item/{item_id}')
def get_item(item_id: int = Path(None, description = 'The ID of the item you would like to use')):
    return inventory[item_id]

# query parameter with "?"
# eg: facebook.com/home?redirect=/nam
from typing import Optional

@app.get("/get-by-name/{item_id}")
def get_item(name: Optional[str] = None, test: int, item_id: int):
    for item_id in inventory:
        if inventory[item_id]['name'] == name:
            return inventory[item_id]
    return {"Data": "Not found"}

# >> localhost.8000/get-by-name/1?test=2&name=Milk

# POST
from pydantic import BaseModel
class Item(BaseModel):
    name: str
    price: float
    brand: Optional[str] = None

@app.post("/create-item/{item_id}")
def create_item(item_id: int, item: Item):
    if item_id in inventory:
        return {"Error": "Item ID already exists."}

    inventory[item_id] = {"name": item.name, "brand": item.brand, "price": :item.price}

    return inventory[item_id]

# PUT
class UpdateItem(BaseModel):
    name: Optional[str] = None
    price: Optional[float] = None
    brand: Optional[str] = None

@app.put("/update-item/{item_id}")
def update_item(item_id: int, item: UpdateItem):
    if item_id not in inventory:
        return {"Error": "Item ID does not exisits"}

    if item.name != None:
        inventory[item_id].name = item.name

    if item.price != None:
        inventory[item_id].price = item.price

    if item.brand != None:
        inventory[item_id].brand = item.brand

    return inventory[item_id]

# DELETE
@app.delete("/delete-item")
def delete_item(item_id: int = Query(...,
                description = "The ID of the itm to delete")
):
    if item_id not in inventory:
        return {"Error": "ID does not exist"}

    del inventory[item_id]
    return {"Success": "Item deleted!"}

# status_code
# eg
from fastapi import HTTPException, status
raise HTTPException(status_code = 404, detail = "Item name not found ")
