from src.domains.detect.detector import VideoDetectorEnum, ImageDetectorEnum
from wtforms import SelectField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.fields.simple import SubmitField, HiddenField


class UploadImageForm(FlaskForm):
    image = FileField(
        validators=[
            FileRequired("Image file is required"),
            FileAllowed(
                ["jpg", "png", "jpeg", "gif", "webp", "bmp"],
                "지원하지 않는 파일입니다.",
            ),
        ]
    )
    submit = SubmitField("감지 시작")


class UploadVideoForm(FlaskForm):
    video = FileField(
        validators=[
            FileRequired("Video file is required"),
            FileAllowed(["mp4", "avi", "mkv"], "지원하지 않는 파일입니다."),
        ]
    )
    submit = SubmitField("Upload")


class DetectImageForm(FlaskForm):
    image_id = HiddenField(validators=[DataRequired()])
    model = SelectField(
        "모델 선택",
        choices=[(de.value, de.name) for de in ImageDetectorEnum],
        default=ImageDetectorEnum.FireDetectV1.value,
    )


class DetectVideoForm(FlaskForm):
    video_id = HiddenField(validators=[DataRequired()])
    model = SelectField(
        "모델 선택",
        choices=[(de.value, de.name) for de in VideoDetectorEnum],
        default=VideoDetectorEnum.FireDetectV1.value,
    )

# 삭제 기능을 위한 폼 (CSRF 토큰 제공 목적)
class DeleteImageForm(FlaskForm):
    submit = SubmitField("삭제")
