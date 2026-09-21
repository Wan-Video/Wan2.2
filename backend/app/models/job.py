import uuid, datetime
from sqlalchemy import Column, String, Float, DateTime, Text, Integer
from app.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default="queued")  # queued | processing | completed | failed
    model = Column(String, default="ti2v-5b")
    prompt = Column(Text)
    image_path = Column(String, nullable=True)
    steps = Column(Integer, default=30)
    guidance_scale = Column(Float, default=6.0)
    seed = Column(Integer, nullable=True)
    size = Column(String, default="704*1280")
    output_path = Column(String, nullable=True)
    progress = Column(Float, default=0.0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
