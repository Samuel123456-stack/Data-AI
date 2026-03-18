from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# 1. Criar uma chain para pegar um texto em outra língua para o português.
# 2. Criar uma chain para resumir em texto
# 3. Criar uma chain que combine a chain 1 e 2

agent = ChatOpenAI(model='gpt-5-nano')

def translator():
    template = ChatPromptTemplate.from_template(
        """
        Você é um tradutor especializado.
        Traduza a frase inteira a seguir para o idioma em português brasileiro
        de forma natural e fluida.
        {text}
        """
    )

    return template | agent | StrOutputParser()

def summary():
    template = ChatPromptTemplate.from_template(
        """
        Você é especialista em resumos.
        Resuma o texto abaixo de forma clara e concisa, mantendo os pontos importantes.
        {text}
        """
    )

    return template | agent | StrOutputParser()

def createTranslationAndSummary():
    translation_chain = translator()
    summary_chain = summary()

    combined_chain = (
        translation_chain
        | (lambda t_text: {'text': t_text})
        | summary_chain
    )
    
    return combined_chain

if __name__ == '__main__':
    sample_text = """
    Artificial intelligence is transforming the way we work and live. 
    Machine learning models are becoming increasingly sophisticated, 
    enabling computers to perform tasks that once required human intelligence. 
    From healthcare to finance, AI applications are revolutionizing industries 
    and creating new opportunities for innovation and efficiency.
    """

    print('-'*10, 'TRADUÇÃO DO TEXTO EM PORTUGUÊS', '-'*10)
    translate_text = translator()
    translated = translate_text.invoke({'text': sample_text})
    print(translated)
    print()

    print('-'*10, 'GERA RESUMO DO TEXTO', '-'*10)
    summary_text = summary()
    summarizer = summary_text.invoke({'text': sample_text})
    print(summarizer)
    print()

    print('-'*10, 'COMBINAÇÃO DA CHAIN 1 E 2', '-'*10)
    combination = createTranslationAndSummary()
    combined = combination.invoke({'text': sample_text})
    print(combined)