from rest_framework.response import Response

class MyResponse(Response):
    headers={"Access-Control-Max-Age":"2592000"}