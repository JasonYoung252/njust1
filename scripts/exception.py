class OCRException(Exception):
    """
    OCR处理过程中发生的异常。

    Attributes:
        message (str): 错误描述
        error_code (int): 错误码
    """

    def __init__(self, message: str, error_code: int = 1001):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class CameraException(Exception):
    """
    拍照或摄像头操作失败时抛出的异常。

    Attributes:
        message (str): 错误描述
        error_code (int): 错误码
    """

    def __init__(self, message: str, error_code: int = 2001):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class RobotArmMoveException(Exception):
    """
    机械臂移动失败时抛出的异常。

    Attributes:
        message (str): 错误描述
        error_code (int): 错误码
    """

    def __init__(self, message: str, error_code: int = 3001):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class NoButtonException(Exception):
    """
    视觉检测不到按钮
    """

    def __init__(self, message: str, error_code: int = 3001):
        self.message = message
        self.error_code = error_code
        super().__init__(message)