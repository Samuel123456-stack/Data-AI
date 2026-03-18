# ChatModels são um componente central do LangChain.

# Um modelo de chat é um modelo de linguagem que utiliza mensagens de chat como entradas
# e retorna mensagens de chat como saidas (ao invés de usar texto puro).

# O LangChain possui integrações com vários provedores (OpenAI, Cohere, Hugging Face, etc) e expõe
# uma interface padrão para interagir com todos esses modelos.

# ----------------------------------------------------------------------------

# Existem 5 tipos diferentes de mensagens:

# 1. __HumanMessage__: Isso representa uma mensagem do usuário. Geralmente consiste apenas do conteúdo
# 2. __AIMessage__: Isso representa uma mensagem do modelo. Pode ter `additional_kwargs` incluidos - por exemplo: `tool_calls()`, se estiver usando chamadas de ferramentas da OpenAI
# 3. __SystemMessage__: Isso representa uma mensagem do sistema, que indica ao modelo como se comportar. Geralmente consiste apenas do conteúdo. Nem todo mundo suporta isso.
# 4. __FunctionMessage__: Isso representa o resultado de uma chamada de função. Além do papel e conteúdo, esta mensagem tem um parâmetro de nome que transmite o nome da função que foi chamada para produzir este resultado.
# 5. __ToolMessage__: Isso representa o resultado de uma chamada de ferramenta. Isso é distinto de uma *Mensagem de Função* a fim de corresponder aos tipos de mensagens de função e ferramentas da OpenAI. Além do papel e conteúdo, esta mensagem tem um parâmetro `tool_call_id()` que transmite o **id** da chamada à ferramenta que foi feita para produzir este resultado.

# 6. __Prompt Templates__: Um prompt para um modelo de linguagem é um conjunto de instruções ou entradas fornecidas por um usuário para guiar a resposta do modelo, ajudando-o a entender o contexto e gerar uma saída baseada em linguagem relevante e coerente, como responder a perguntas, completar frases ou participar de uma conversa.

# 7. __Output Parsers - Formatando Saídas__: Como retornar dados estruturados de um modelo? É frequentemente útil que um modelo retorne uma saída que corresponda a um esquema específico. Um caso de uso comum é a extração de dados de um texto para inseri-los em um banco de dados ou utilizá-los em algum outro sistema subsequente. Nesta aula abordaremos algumas estratégias para obter saídas estruturadas de um modelo.

# 8. __Estruturando saidas de chat - [StrOutputParser]__: O formatador mais simples do LangChain é o *StrOutputParser. Ele é utilizado para convertermos saídas do modelo no formato de conversação para formato texto. É um atividade bem comum, levando em consideração que maior parte das llms que utilizamos com LangChain são acessadas através dos ChatModels.

## ----------------------------------------------------------------------------------------------------

# *LANGCHAIN EXPRESSION LANGUAGE (LCEL)*

# A Linguagem de Expressão LangChain, ou LCEL, é uma forma declarativa de **compor cadeias de maneira fácil**. A LCEL foi projetada desde o primeiro dia para suportar a colocação de protótipos em produção, sem a necessidade de alterações no código, desde a cadeia mais simples “prompt + LLM” até as cadeias mais complexas (já vimos pessoas executando com sucesso cadeias LCEL com centenas de etapas em produção). Para destacar alguns dos motivos pelos quais você pode querer usar a LCEL:

- **Suporte a streaming de primeira classe**: menor tempo possível para saída do primeiro token produzido;
- **Suporte assíncrono**: Qualquer cadeia construída com a LCEL pode ser chamada tanto com a API síncrona;
- **Execução paralela otimizada**: Sempre que suas cadeias LCEL tiverem etapas que podem ser executadas em paralelo, automaticamente é feito isso;
- **Retentativas e fallbacks**: É maneira de tornar suas cadeias mais confiáveis em grande escala, na qual ações alternativas podem ser tomadas no caso de um erro em uma cadeia
- **Acessar resultados intermediários**: auxiliando na depuração de uma cadeia;

## ----------------------------------------------------------------------------------------------------

# *Estruturando saídas mais complexas - Pydantic*
# Utilizando `with_structured_output()`

# Esta é a maneira mais fácil e confiável de obter saídas estruturadas. O método with_structured_output() é implementado para modelos que fornecem APIs nativas para estruturar saídas, como chamadas de ferramentas/funções ou modo JSON, e aproveita essas capacidades internamente.

# Este método recebe um esquema como entrada, que especifica os nomes, tipos e descrições dos atributos desejados na saída. Ele retorna um objeto similar a um Runnable, exceto que, em vez de gerar strings ou mensagens, produz objetos correspondentes ao esquema fornecido. O esquema pode ser especificado como uma classe TypedDict, um JSON Schema ou uma classe Pydantic.

## ----------------------------------------------------------------------------------------------------

# *Embeddings*

# Os [Embeddings] cria uma representação vetorial de um pedaço de texto. Isso é útil porque significa que podemos pensar sobre o texto no espaço vetorial e fazer coisas como busca semântica, onde procuramos por pedaços de texto que são mais semelhantes no espaço vetorial, ou seja, que estão a uma distância menor.

# A classe [Embeddings] do Langchain é uma classe projetada para interagir com modelos de embedding de texto. Existem muitos modelos diferentes (OpenAI, Cohere, Hugging Face, etc). E ela fornece uma interface padrão para todos eles.

# A classe de [Embeddings] fornece dois métodos: **realizar o embedding de documentos** e **realizar embedding de uma chamada**. O primeiro recebe como entrada vários textos, enquanto o último recebe um único texto.

## ----------------------------------------------------------------------------------------------------

# *VectorStores*

# Uma das maneiras mais comuns de armazenar e buscar dados não estruturados é realizando o embedding e armazenando vetores resultantes e, em seguida, na hora da consulta, realizar o embedding da consulta e recuperar os vetores `mais semelhantes`.

## ----------------------------------------------------------------------------------------------------

# *Descrição dos arquivos*

# [Fundamentos_e_Prompts]

# 1. __chat_models.py__: Instancia um modelo OpenAI e simula uma conversa básica com mensagens de sistema e humano.

# 2. __prompt_templates.py__: Demonstra o uso de `PromptTemplate` e `ChatPromptTemplate` com diferentes configurações de instruções.

# [Chains_LCEL]

# 1. __chains_lcel.py__: Cria chains simples com Langchain Expression Language com e sem parser de saída.

# 2. __chain_lcel_challenge.py__: Combina duas chains: uma tradução e outra de resumo, encadeadas em sequência.

# 3. __route_chain.py__: Implementa um roteador que categoriza perguntas e as direciona para agentes especializados (matemática, física ou história).

# [Leitura_Documentos]

# 1. __pdf_reader.py__: Lê arquivos PDF e responde via cadeia de Q&A
# 2. __csv_reader.py__: Lê arquivo CSV e responde perguntas com base no conteúdo.
# 3. __api_reader.py__: Carrega conteúdo de página Web via URL.
# 4. __custom_loader.py__: Loader customizado para carregar documentos de fontes personalizadas.

# [Embeddings_Vetores]

# 1. __embedding.py__: Gera embeddings de textos e calcula coeficiente de similaridade entre eles.
# 2. __vectorstore.py__: Cria e consulta um banco vetorial ChromaDB com busca por similaridade.
# 3. __retrieve.py__: Similar ao anterior, mas com suporte a múltiplos documentos e metadados.

# [RAG_Memória]

# 1. __rag.py__: Pipeline completo de RAG (Retrieve-Augmented Generation) com múltiplos PDFs e prompt customizado.
# 2. __memory.py__: Chatbot com memória de sessão usando `InMemoryChatMessageHistory`

# [Output_Parser]

# 1. __output_parsers.py__: Demonstra o uso de `StrOutputParser` para formatar respostas.
# 2. __pydantic_output_parsers.py__: Usa `Pydantic` para estruturar saídas do modelo em objetos tipados.