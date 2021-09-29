from pykospacing import Spacing

import torch
import joblib

from konlpy.tag import Mecab
from model.kobert import KoBERTClassifier
from model.kogpt2 import KoGPT2Replier


class model_loader():
    def __init__(self):
        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(ctx)

        # Spacing
        self.spacing = Spacing()
        
        # Mecab
        self.mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

        # koBERT
        self.label_encoder = joblib.load('saves/labelencoder.pkl')

        checkpoint = torch.load('saves/kobert_model.pth', map_location=self.device)
        self.classifier = KoBERTClassifier()
        self.classifier.load_state_dict(checkpoint, strict=False)
        self.classifier.eval()

        # koGPT2
        checkpoint = torch.load("saves/kogpt2_model.pth", map_location=self.device)
        self.generator = KoGPT2Replier()
        self.generator.load_state_dict(checkpoint, strict=False)
        self.generator.eval()

    # 문장 띄어쓰기 적용
    def split_msg(self, msg):
        return self.spacing(msg)

    # 단어 추출
    def tokenize_msg(self, msg):
        keywords = [word for word, tag in self.mecab.pos(msg) if tag in ('NNG', 'NNP', 'VV', 'VA')]
        return keywords

    # 감정 분류
    def classify_msg(self, msg):
        category = self.classifier.classify(msg=msg, label_encoder=self.label_encoder)
        return category

    # 답변 생성
    def generate_reply(self, msg):
        reply = self.generator.reply(msg=msg)
        return reply

    

    