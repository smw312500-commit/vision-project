import datetime
from enum import Enum
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.main import db


class UserImage(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    image_path: Mapped[str] = mapped_column()
    created_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC),
        onupdate=datetime.datetime.now(datetime.UTC),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    user: Mapped["User"] = relationship(back_populates="images")

    detection_images: Mapped[List["DetectionImage"]] = relationship(
        back_populates="user_image"
    )


class UserVideo(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    video_path: Mapped[str] = mapped_column()
    thumbnail_path: Mapped[str] = mapped_column(nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC),
        onupdate=datetime.datetime.now(datetime.UTC),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    user: Mapped["User"] = relationship(back_populates="videos")

    detection_videos: Mapped[List["DetectionVideo"]] = relationship(
        back_populates="user_video"
    )


class DetectionImage(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column()
    image_path: Mapped[str] = mapped_column()

    user_image_id: Mapped[int] = mapped_column(ForeignKey("user_image.id"))
    user_image: Mapped["UserImage"] = relationship(back_populates="detection_images")


class TaskStatus(Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    FAILURE = "FAILURE"
    SUCCESS = "SUCCESS"


class DetectionVideo(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column()
    video_path: Mapped[str] = mapped_column(nullable=True)
    status: Mapped[TaskStatus] = mapped_column(default=TaskStatus.PENDING)

    user_video_id: Mapped[int] = mapped_column(ForeignKey("user_video.id"))
    user_video: Mapped["UserVideo"] = relationship(back_populates="detection_videos")
