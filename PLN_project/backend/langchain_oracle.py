import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from typing import Dict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.8, model="gpt-4o-mini")

class OraclePipeline:
    def __init__(self):
        self.combine_template = PromptTemplate(
            input_variables=["user", "wiki_signo", "wiki_time", "wiki_cidade", "artist_info"],
            template=(
                "Você é um oráculo poético. Recebeu as seguintes informações (texto bruto) e deve produzir saídas intermediárias.\n\n"
                "Usuário:\n{user}\n\n"
                "Resumo - Signo:\n{wiki_signo}\n\n"
                "Resumo - Time:\n{wiki_time}\n\n"
                "Resumo - Cidade:\n{wiki_cidade}\n\n"
                "Info - Artista (Spotify):\n{artist_info}\n\n"
                "Combine tudo num único documento coerente, mantendo citações das partes originais (cite as fontes em linhas com 'Fonte: ...')."
            )
        )
        self.combine_chain = self.combine_template | llm

        self.ner_template = PromptTemplate(
            input_variables=["combined_doc"],
            template=(
                "Extraia as ENTIDADES NOMEADAS do texto abaixo em formato JSON com chaves: Pessoas, Locais, Organizações, Eventos, Outros. "
                "Para cada entidade inclua: 'texto' e 'origem' (ex: Signo, Time, Cidade, Artista, Usuário). "
                "Se não houver itens para uma categoria, retorne lista vazia.\n\n"
                "Texto:\n{combined_doc}\n\n"
                "Saída JSON estrita (apenas JSON):"
            )
        )
        self.ner_chain = self.ner_template | llm

        self.keywords_template = PromptTemplate(
            input_variables=["combined_doc"],
            template=(
                "Extraia as 12 palavras-chave (ou frases curtas) mais relevantes do texto abaixo. "
                "Retorne um JSON com chave 'keywords' que é uma lista ordenada da mais relevante para a menos.\n\n"
                "Texto:\n{combined_doc}\n\n"
                "Saída JSON:"
            )
        )
        self.keywords_chain = self.keywords_template | llm

        self.classif_template = PromptTemplate(
            input_variables=["combined_doc"],
            template=(
                "Classifique o perfil em três categorias inventadas e criativas baseadas no texto:\n"
                "- personalidade_musical\n"
                "- vibe_futebolistica\n"
                "- polaridade_emocional\n\n"
                "Dê também uma explicação curta (1-2 frases) para cada classificação. Retorne JSON.\n\n"
                "Texto:\n{combined_doc}\n\n"
                "Saída JSON:"
            )
        )
        self.classif_chain = self.classif_template | llm

        self.prediction_template = PromptTemplate(
            input_variables=["combined_doc", "keywords_json", "ner_json", "classification_json", "user"],
            template=(
                "Você é o Oráculo Estocástico: gere uma PREVISÃO FINAL fictícia, mística, humorística e poética para o usuário. "
                "Use todas as informações a seguir: documento combinado, entidades extraídas, palavras-chave e classificações. "
                "A previsão deve:\n"
                "- ser claramente sinalizada como IMAGINÁRIA e para entretenimento\n"
                "- misturar referências reais (citadas com 'Fonte: ...') com elementos absurdos\n"
                "- ter entre 6 e 12 linhas curtas\n"
                "- terminar com uma 'dica prática' divertida (1 frase)\n\n"
                "Forneça também um pequeno parágrafo (2-3 frases) explicando como a previsão se relaciona às informações reais.\n\n"
                "Document:\n{combined_doc}\n\nNer JSON:\n{ner_json}\n\nKeywords JSON:\n{keywords_json}\n\nClassificação JSON:\n{classification_json}\n\nUser:\n{user}\n\nSaída JSON com chaves: prediction (string com quebras de linha), explanation (string)."
            )
        )
        self.pred_chain = self.prediction_template | llm

    def run_pipeline(self, inputs: Dict):
        combined = self.combine_chain.invoke({
            "user": inputs["user"],
            "wiki_signo": inputs["wiki_signo"],
            "wiki_time": inputs["wiki_time"],
            "wiki_cidade": inputs["wiki_cidade"],
            "artist_info": inputs["artist_info"]
        }).content

        ner = self.ner_chain.invoke({"combined_doc": combined}).content
        kws = self.keywords_chain.invoke({"combined_doc": combined}).content
        cls = self.classif_chain.invoke({"combined_doc": combined}).content
        pred = self.pred_chain.invoke({
            "combined_doc": combined,
            "keywords_json": kws,
            "ner_json": ner,
            "classification_json": cls,
            "user": inputs["user"]
        }).content

        return {
            "combined_doc": combined,
            "ner_json": ner,
            "keywords_json": kws,
            "classification_json": cls,
            "prediction_json": pred
        }