from rest_framework.throttling import AnonRateThrottle, UserRateThrottle

class AnonWriteRateThrottle(AnonRateThrottle):
    THROTTLE_RATES = {"anon": "5/min"}

class AnonReadRateThrottle(AnonRateThrottle):
    THROTTLE_RATES = {"anon": "10/min"}

class UserWriteRateThrottle(UserRateThrottle):
    THROTTLE_RATES = {"user": "15/min"}

class UserReadRateThrottle(UserRateThrottle):
    THROTTLE_RATES = {"user": "30/min"}

class UserAIRateThrottle(UserRateThrottle):
    THROTTLE_RATES = {"user": "2/min"}