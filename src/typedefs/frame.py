from pydantic import BaseModel, ConfigDict

class Frame(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    sequence_id: str
    frame_id: int
    rgb_image: str
    rgb_label: str
    ir_image: str
    ir_label: str

    def __str__(self) -> str:
        """Custom structural string representation for clean logging."""
        # Zero-pad the frame ID dynamically to 5 digits for structural alignment
        padded_frame = f"{self.frame_id:05}"
        
        return (
            f"Frame Entity [{self.sequence_id} | #{padded_frame}]\n"
            f"├── RGB Image : {self.rgb_image}\n"
            f"├── RGB Label : {self.rgb_label}\n"
            f"├── IR Image  : {self.ir_image}\n"
            f"└── IR Label  : {self.ir_label}"
        )