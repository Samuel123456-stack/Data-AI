const API_URL = "http://127.0.0.1:8000/recommendations";

const form = document.querySelector("#recommendation-form");
const textarea = document.querySelector("#preferences");
const characterCount = document.querySelector("#character-count");
const submitButton = document.querySelector("#submit-button");
const emptyState = document.querySelector("#empty-state");
const loadingState = document.querySelector("#loading-state");
const errorState = document.querySelector("#error-state");
const errorMessage = document.querySelector("#error-message");
const movieGrid = document.querySelector("#movie-grid");
const movieCount = document.querySelector("#movie-count");
const resultsSummary = document.querySelector("#results-summary");
const template = document.querySelector("#movie-card-template");
const apiStatus = document.querySelector("#api-status");
const apiStatusText = document.querySelector("#api-status-text");

function setIconState() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function updateCharacterCount() {
  const count = textarea.value.trim().length;
  characterCount.textContent = `${count}/100`;
}

function setState(state, message = "") {
  emptyState.classList.toggle("hidden", state !== "empty");
  loadingState.classList.toggle("hidden", state !== "loading");
  errorState.classList.toggle("hidden", state !== "error");
  movieGrid.classList.toggle("hidden", state !== "success");
  errorMessage.textContent = message;

  apiStatus.classList.toggle("loading", state === "loading");
  apiStatus.classList.toggle("ok", state === "success");
  apiStatusText.textContent = state === "success" ? "API ativa" : state === "loading" ? "Consultando API" : "API aguardando";
}

function formatList(items, fallback) {
  return Array.isArray(items) && items.length ? items.join(", ") : fallback;
}

function renderMovies(movies) {
  movieGrid.innerHTML = "";

  movies.forEach((movie) => {
    const card = template.content.cloneNode(true);
    const poster = card.querySelector(".poster");
    const title = card.querySelector("h3");
    const year = card.querySelector(".year");
    const meta = card.querySelector(".meta");
    const genres = card.querySelector(".genres");
    const synopsis = card.querySelector(".synopsis");
    const reason = card.querySelector(".reason");
    const rating = card.querySelector(".rating");
    const director = card.querySelector(".director");
    const platforms = card.querySelector(".platforms");

    title.textContent = movie.title || "Filme recomendado";
    year.textContent = movie.release_year || "";
    meta.textContent = `${movie.duration_minutes || "--"} min · ${movie.primary_language || "Idioma não informado"} · ${movie.age_rating || "Livre"}`;
    synopsis.textContent = movie.synopsis || "Sinopse não informada.";
    reason.textContent = movie.recommendation_reason || "Boa escolha para as preferências descritas.";
    rating.textContent = `IMDb ${movie.imdb_rating ?? "--"}`;
    director.textContent = `Direção: ${movie.director || "Não informada"}`;
    platforms.textContent = `Onde assistir: ${formatList(movie.streaming_platforms, "Não informado")}`;

    if (movie.poster_url) {
      poster.src = movie.poster_url;
      poster.alt = `Poster de ${movie.title}`;
      poster.onerror = () => {
        poster.removeAttribute("src");
        poster.alt = "";
      };
    } else {
      poster.removeAttribute("src");
      poster.alt = "";
    }

    (movie.genres || []).slice(0, 4).forEach((genre) => {
      const pill = document.createElement("span");
      pill.textContent = genre;
      genres.appendChild(pill);
    });

    movieGrid.appendChild(card);
  });

  movieCount.textContent = `${movies.length} ${movies.length === 1 ? "filme" : "filmes"}`;
  resultsSummary.textContent = movies.length
    ? "Cards gerados a partir da resposta da API."
    : "A API respondeu, mas não retornou filmes.";
  setIconState();
}

async function requestRecommendations(preferences) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ preferences }),
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const detail = payload?.detail;
    throw new Error(typeof detail === "string" ? detail : "Não foi possível gerar recomendações.");
  }

  return payload;
}

textarea.addEventListener("input", updateCharacterCount);

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const preferences = textarea.value.trim();
  if (preferences.length < 50 || preferences.length > 100) {
    setState("error", "Descreva suas preferências com 50 a 100 caracteres para respeitar o contrato da API.");
    return;
  }

  submitButton.disabled = true;
  setState("loading");

  try {
    const payload = await requestRecommendations(preferences);
    const movies = payload?.data?.movies || [];
    renderMovies(movies);
    setState("success");
  } catch (error) {
    setState("error", error.message);
  } finally {
    submitButton.disabled = false;
  }
});

updateCharacterCount();
setState("empty");
setIconState();
