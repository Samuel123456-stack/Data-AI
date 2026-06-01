# FilmPro

FilmPro é uma aplicação de recomendação inteligente de filmes em português. O projeto combina uma API FastAPI, um agente de IA com OpenAI, busca web, consulta ao TMDB e uma interface web estática para transformar preferências do usuário em cards de filmes recomendados.

## Objetivo

O objetivo do FilmPro é permitir que uma pessoa descreva rapidamente o tipo de filme que quer assistir e receba uma lista estruturada de recomendações. Cada recomendação pode incluir título, ano, diretor, gêneros, nota IMDb, duração, idioma, sinopse, classificação indicativa, elenco, pôster, plataformas de streaming e o motivo da indicação.

## Escopo da aplicação

O projeto está organizado em quatro partes principais:

- `api`: camada HTTP responsável por expor a aplicação para clientes externos.
- `agent`: núcleo inteligente que interpreta preferências, pesquisa filmes e estrutura as recomendações.
- `site`: interface gráfica usada pelo usuário final para enviar preferências e visualizar os resultados.
- `design`: sistema visual de referência usado para criar a interface do site.

## Estrutura de diretórios

```text
FilmPro/
|-- agent/
|   |-- config.py              # Leitura e validação das variáveis de ambiente
|   |-- core.py                # Configuração do agente FilmPro e função recommend()
|   |-- models/
|   |   `-- movies.py          # Modelos Pydantic da resposta de filmes
|   |-- prompts/
|   |   `-- movie_search.py    # Descrição e instruções do agente
|   `-- tools/
|       |-- tmdb.py            # Ferramenta assíncrona de consulta ao TMDB
|-- api/
|   |-- app.py                 # Instância FastAPI, CORS e inclusão de rotas
|   `-- routers.py             # Endpoint POST /recommendations
|-- design/
|   `-- ai-social-automation.aura.build/
|       |-- design-system.html # Referência visual do design system
|       |-- index.html         # Página de exemplo do design
|       `-- assets/            # Fontes, ícones e assets do design
|-- site/
|   |-- index.html             # Interface principal da aplicação
|   |-- styles.css             # Estilos inspirados no design system
|   `-- app.js                 # Integração frontend com a API
|-- main.py                    # Entrada para executar o servidor Uvicorn
|-- pyproject.toml             # Dependências e metadados do projeto
|-- requirements.txt           # Dependência mínima para uso com uv
`-- uv.lock                    # Lockfile de dependências
```

## Fluxo de funcionamento

1. O usuário abre a interface em `site/index.html`.
2. Ele escreve suas preferências de filme em um campo de texto.
3. O frontend envia uma requisição `POST` para `http://127.0.0.1:8000/recommendations`.
4. A API valida o corpo da requisição e chama `agent.core.recommend()`.
5. O agente FilmPro usa o modelo OpenAI, busca web e a ferramenta TMDB para montar recomendações.
6. A resposta é validada pelos modelos Pydantic em `agent/models/movies.py`.
7. O frontend recebe `data.movies` e renderiza cards com as recomendações.

## Contrato da API

### `POST /recommendations`

Gera recomendações personalizadas com base nas preferências do usuário.

Corpo da requisição:

```json
{
  "preferences": "Quero um suspense inteligente, com ritmo tenso e final surpreendente."
}
```

Regras do campo `preferences`:

- Mínimo: 50 caracteres.
- Máximo: 100 caracteres.
- Deve descrever preferências, clima, gênero, tema ou referências de filmes.

Resposta esperada:

```json
{
  "success": true,
  "data": {
    "movies": [
      {
        "title": "Movie title",
        "release_year": 2020,
        "director": "Director name",
        "genres": ["Suspense", "Drama"],
        "imdb_rating": 8.1,
        "duration_minutes": 120,
        "primary_language": "Inglês",
        "synopsis": "Sinopse breve do filme.",
        "age_rating": "16",
        "content_warnings": ["Violência"],
        "cast": [
          {
            "name": "Actor name",
            "character": "Character name"
          }
        ],
        "poster_url": "https://...",
        "streaming_platforms": ["Netflix"],
        "recommendation_reason": "Motivo da recomendação."
      }
    ],
    "total_recommendations": 1
  },
  "message": "Recomendações geradas com sucesso."
}
```

## Interface web

A interface em `site` é uma página estática composta por:

- Campo centralizado para descrever preferências de filmes.
- Botão para iniciar o fluxo com a API.
- Indicador de status no canto superior direito.
- Estados visuais de espera, carregamento, erro e sucesso.
- Cards de filme gerados a partir da resposta da API.

O indicador de status muda para `API ativa` com luz verde quando a API retorna resultado com sucesso.

## Design system

O diretório `design/ai-social-automation.aura.build` contém a referência visual usada no site. A interface atual reaproveita a linguagem desse design system:

- Fundo escuro.
- Gradientes e brilho em laranja.
- Tipografia com Inter e Bricolage Grotesque.
- Ícones Lucide.
- Cards com bordas discretas e realce luminoso.

## Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto com:

```env
OPENAI_API_KEY=sua_chave_openai
TMDB_API_KEY=sua_chave_tmdb
```

Essas variáveis são obrigatórias. A aplicação chama `Config.validate()` durante a inicialização e interrompe a execução se alguma chave estiver ausente. `TMDB_API_KEY` é opcional e fica apenas como suporte para a ferramenta legada `agent/tools/tmdb.py`.

## Como executar

### 1. Preparar o ambiente

Com `uv`:

```bash
uv sync
```

Ou com `pip`:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Configurar o `.env`

Adicione `OPENAI_API_KEY` e `TMDB_API_KEY` na raiz do projeto.

### 3. Subir a API

```bash
python main.py
```

A API ficará disponível em:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

### 4. Abrir o site

Abra o arquivo:

```text
site/index.html
```

O site envia requisições para `http://127.0.0.1:8000/recommendations`.

## Principais tecnologias

- Python
- FastAPI
- Pydantic
- Uvicorn
- Agno
- OpenAI
- DuckDuckGo/WebSearchTools
- TMDB API
- HTML, CSS e JavaScript

## Observações de desenvolvimento

- O backend permite CORS para facilitar o consumo pela interface estática.
- O agente retorna uma estrutura tipada por `MovieRecommendation`.
- O frontend espera que a resposta tenha filmes em `data.movies`.
- O contrato atual do endpoint limita o texto de preferências entre 50 e 100 caracteres.
- O projeto usa `main.py` como ponto de entrada do servidor.
