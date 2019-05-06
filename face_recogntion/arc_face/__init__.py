from ctypes import *

APP_ID = b'8MJMLoy59856HJ4yqcygJQmzopyPsUcY2xSnd2e3hRq3'
SDK_KEY = b'B7D9jikEEzt5c9DDNtCe2scBxKBqWfQr7h3pWWCfZwbR'


# 人脸框
class MRECT(Structure):
    _fields_ = [(u'left', c_int32), (u'top', c_int32), (u'right', c_int32), (u'bottom', c_int32)]
