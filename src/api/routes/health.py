from fastapi import APIRouter

router = APIRouter()


@router.get("/check")
def health():
    pass
