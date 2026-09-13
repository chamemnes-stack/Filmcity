import math
import requests
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION STREAMLIT
# ==================================================

st.set_page_config(
    page_title="FilmCity",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# STYLE CSS
# ==================================================

st.markdown(
    """
    <style>

    html,
    body,
    [data-testid="stAppViewContainer"],
    .stApp {
        background:
            radial-gradient(
                circle at top left,
                #182035 0%,
                #0D111B 35%,
                #080B12 100%
            ) !important;

        color: #F7F7F7 !important;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }

    .main {
        background: transparent !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    /* TITRES */

    .hero-title {
        font-size: 3.3rem;
        font-weight: 800;
        margin-bottom: 0;
        color: white;
    }

    .hero-subtitle {
        font-size: 1.35rem;
        font-weight: 500;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        color: #D8DCE7;
    }


    /* TEXTES */

    p,
    span,
    label {
        color: #F4F4F4;
    }


    /* BARRE DE RECHERCHE */

    div[data-baseweb="input"] {
        background-color: #171D2A !important;
        border-radius: 12px !important;
        border: 1px solid #30384A !important;
    }

    div[data-baseweb="input"] input {
        color: white !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #9098A9 !important;
    }


    /* BOUTONS */

    div.stButton > button {
        background-color: #151B28 !important;
        color: white !important;
        border: 1px solid #2D3547 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #FF6584 !important;
        transform: translateY(-1px);
    }


    /* MODES */

    div[role="radiogroup"] {
        gap: 12px;
    }

    div[role="radiogroup"] label {
        background-color: #151B28;
        border: 1px solid #2D3547;
        padding: 9px 14px;
        border-radius: 12px;
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] label:hover {
        transform: translateY(-2px);
        border-color: #7581A0;
    }


    /* BADGES */

    .mode-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 12px;
    }

    .mode-similaire {
        background: rgba(255, 83, 120, 0.16);
        color: #FF7996;
        border: 1px solid rgba(255, 83, 120, 0.45);
    }

    .mode-qualite {
        background: rgba(255, 193, 7, 0.14);
        color: #FFD35A;
        border: 1px solid rgba(255, 193, 7, 0.40);
    }

    .mode-decouverte {
        background: rgba(134, 108, 255, 0.16);
        color: #B7A7FF;
        border: 1px solid rgba(134, 108, 255, 0.45);
    }


    /* POSTERS */

    [data-testid="stImage"] img {
        border-radius: 12px;
        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }

    [data-testid="stImage"] img:hover {
        transform: scale(1.025);

        box-shadow:
            0 10px 30px
            rgba(0, 0, 0, 0.45);
    }


    /* FILMS */

    .film-title {
        color: white;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 7px;
        min-height: 42px;
    }

    .film-info {
        color: #AEB6C4;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 12px;
    }


    /* EXPANDER */

    details {
        background-color:
            rgba(20, 25, 36, 0.70) !important;

        border:
            1px solid
            rgba(255, 255, 255, 0.08) !important;

        border-radius: 10px !important;
    }

    details summary {
        color: white !important;
    }


    /* MOBILE */

    @media (max-width: 800px) {

        .hero-title {
            font-size: 2.5rem;
        }

        .hero-subtitle {
            font-size: 1.1rem;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# CHARGEMENT DES DONNEES
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "filmcity_recommendation.csv"
    )

    df = df.dropna(
        subset=[
            "primaryTitle",
            "genres",
            "synopsis",
            "weighted_rating",
            "startYear",
            "numVotes",
            "averageRating"
        ]
    ).reset_index(drop=True)

    df["startYear"] = pd.to_numeric(
        df["startYear"],
        errors="coerce"
    )

    df["numVotes"] = pd.to_numeric(
        df["numVotes"],
        errors="coerce"
    )

    df["averageRating"] = pd.to_numeric(
        df["averageRating"],
        errors="coerce"
    )

    df["weighted_rating"] = pd.to_numeric(
        df["weighted_rating"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "startYear",
            "numVotes",
            "averageRating",
            "weighted_rating"
        ]
    ).reset_index(drop=True)

    return df


df_final = load_data()


# ==================================================
# NORMALISATION
# ==================================================

rating_scaler = MinMaxScaler()

df_final["rating_normalized"] = (
    rating_scaler.fit_transform(
        df_final[["weighted_rating"]]
    )
)


popularity_scaler = MinMaxScaler()

df_final["popularity_normalized"] = (
    popularity_scaler.fit_transform(
        df_final[["numVotes"]]
    )
)


# ==================================================
# OUTILS GENRES
# ==================================================

def genre_set(genres):

    if pd.isna(genres):
        return set()

    return {
        genre.strip()
        for genre in str(genres).split(",")
        if genre.strip()
        and genre.strip() != "\\N"
    }


# ==================================================
# POIDS DES GENRES
# ==================================================

@st.cache_data
def calculate_genre_weights(genres_column):

    total_movies = len(
        genres_column
    )

    genre_counts = {}

    for genres in genres_column:

        current_genres = genre_set(
            genres
        )

        for genre in current_genres:

            genre_counts[genre] = (
                genre_counts.get(
                    genre,
                    0
                )
                + 1
            )

    weights = {}

    for genre, count in genre_counts.items():

        weights[genre] = (
            math.log(
                (total_movies + 1)
                /
                (count + 1)
            )
            + 1
        )

    return weights


genre_weights = calculate_genre_weights(
    tuple(
        df_final[
            "genres"
        ].tolist()
    )
)


# ==================================================
# SIMILARITE GENRES
# ==================================================

def genre_similarity(
    genres_a,
    genres_b
):

    set_a = genre_set(
        genres_a
    )

    set_b = genre_set(
        genres_b
    )

    if not set_a or not set_b:
        return 0

    return (
        len(set_a & set_b)
        /
        len(set_a | set_b)
    )


def shared_genres_count(
    genres_a,
    genres_b
):

    set_a = genre_set(
        genres_a
    )

    set_b = genre_set(
        genres_b
    )

    return len(
        set_a & set_b
    )


def weighted_genre_similarity(
    genres_a,
    genres_b
):

    set_a = genre_set(
        genres_a
    )

    set_b = genre_set(
        genres_b
    )

    if not set_a or not set_b:
        return 0

    intersection = (
        set_a & set_b
    )

    union = (
        set_a | set_b
    )

    shared_weight = sum(
        genre_weights.get(
            genre,
            1
        )
        for genre in intersection
    )

    union_weight = sum(
        genre_weights.get(
            genre,
            1
        )
        for genre in union
    )

    if union_weight == 0:
        return 0

    return (
        shared_weight
        /
        union_weight
    )


# ==================================================
# TEXTE SEMANTIQUE
# ==================================================

df_final["semantic_text"] = (
    df_final[
        "synopsis"
    ].fillna("")
)


# ==================================================
# MODELE SEMANTIQUE
# ==================================================

@st.cache_resource
def load_semantic_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


semantic_model = (
    load_semantic_model()
)


# ==================================================
# EMBEDDINGS
# ==================================================

@st.cache_data(
    show_spinner=False
)
def create_embeddings(
    texts
):

    return semantic_model.encode(
        list(texts),
        show_progress_bar=False,
        normalize_embeddings=True
    )


with st.spinner(
    "🎬 Préparation de FilmCity..."
):

    movie_embeddings = (
        create_embeddings(
            tuple(
                df_final[
                    "semantic_text"
                ].tolist()
            )
        )
    )


# ==================================================
# MOTEUR DE RECOMMANDATION
# ==================================================

def recommend_movie(
    title,
    mode="similaire"
):

    matches = df_final[
        df_final[
            "primaryTitle"
        ]
        .str.lower()
        == title.lower()
    ]

    if matches.empty:

        return pd.DataFrame()


    idx = matches.index[0]


    selected_genres = (
        df_final.loc[
            idx,
            "genres"
        ]
    )


    selected_year = float(
        df_final.loc[
            idx,
            "startYear"
        ]
    )


    # --------------------------------------------------
    # SIMILARITE SEMANTIQUE
    # --------------------------------------------------

    selected_embedding = (
        movie_embeddings[
            idx
        ]
    )


    semantic_scores = np.dot(
        movie_embeddings,
        selected_embedding
    )


    recommendations = (
        df_final.copy()
    )


    recommendations[
        "semantic_score"
    ] = semantic_scores


    # --------------------------------------------------
    # RETIRER LE FILM CHOISI
    # --------------------------------------------------

    recommendations = recommendations[
        recommendations.index != idx
    ].copy()


    # --------------------------------------------------
    # GENRES
    # --------------------------------------------------

    recommendations[
        "genre_score"
    ] = recommendations[
        "genres"
    ].apply(
        lambda x:
        genre_similarity(
            selected_genres,
            x
        )
    )


    recommendations[
        "weighted_genre_score"
    ] = recommendations[
        "genres"
    ].apply(
        lambda x:
        weighted_genre_similarity(
            selected_genres,
            x
        )
    )


    recommendations[
        "shared_genres"
    ] = recommendations[
        "genres"
    ].apply(
        lambda x:
        shared_genres_count(
            selected_genres,
            x
        )
    )


    # --------------------------------------------------
    # ANNEE
    # --------------------------------------------------

    recommendations[
        "year_distance"
    ] = (
        recommendations[
            "startYear"
        ]
        - selected_year
    ).abs()


    recommendations[
        "year_score"
    ] = (
        1
        /
        (
            1
            +
            recommendations[
                "year_distance"
            ]
            / 10
        )
    )


    # ==================================================
    # 🎯 MEME VIBE
    # ==================================================

    if mode == "similaire":

        # On garde les films dans une période
        # de maximum 20 ans autour du film choisi.

        recommendations = recommendations[
            recommendations[
                "year_distance"
            ] <= 20
        ].copy()


        # Au moins un genre commun

        recommendations = recommendations[
            recommendations[
                "shared_genres"
            ] >= 1
        ].copy()


        # Similarité sémantique minimum

        recommendations = recommendations[
            recommendations[
                "semantic_score"
            ] >= 0.35
        ].copy()


        recommendations[
            "recommendation_score"
        ] = (
            0.60
            * recommendations[
                "semantic_score"
            ]
            +
            0.25
            * recommendations[
                "weighted_genre_score"
            ]
            +
            0.10
            * recommendations[
                "year_score"
            ]
            +
            0.05
            * recommendations[
                "rating_normalized"
            ]
        )


        n = 10


    # ==================================================
    # ⭐ LES INCONTOURNABLES
    # ==================================================

    elif mode == "qualite":

        # Pas de limite stricte sur l'année.

        recommendations = recommendations[
            recommendations[
                "shared_genres"
            ] >= 1
        ].copy()


        # Film suffisamment populaire

        recommendations = recommendations[
            recommendations[
                "numVotes"
            ] >= 100000
        ].copy()


        # On garde les films les plus proches
        # sémantiquement avant de classer.

        if not recommendations.empty:

            recommendations = (
                recommendations.nlargest(
                    min(
                        200,
                        len(
                            recommendations
                        )
                    ),
                    "semantic_score"
                )
                .copy()
            )


        recommendations[
            "recommendation_score"
        ] = (
            0.30
            * recommendations[
                "semantic_score"
            ]
            +
            0.15
            * recommendations[
                "weighted_genre_score"
            ]
            +
            0.30
            * recommendations[
                "rating_normalized"
            ]
            +
            0.20
            * recommendations[
                "popularity_normalized"
            ]
            +
            0.05
            * recommendations[
                "year_score"
            ]
        )


        n = 15


    # ==================================================
    # 💎 PEPITE A DECOUVRIR
    # ==================================================

    elif mode == "decouverte":

        # Pas de limite stricte sur l'année.

        recommendations = recommendations[
            (
                recommendations[
                    "numVotes"
                ] >= 10000
            )
            &
            (
                recommendations[
                    "numVotes"
                ] <= 300000
            )
            &
            (
                recommendations[
                    "weighted_rating"
                ] >= 6.5
            )
            &
            (
                recommendations[
                    "shared_genres"
                ] >= 1
            )
        ].copy()


        recommendations[
            "recommendation_score"
        ] = (
            0.35
            * recommendations[
                "semantic_score"
            ]
            +
            0.25
            * recommendations[
                "weighted_genre_score"
            ]
            +
            0.30
            * recommendations[
                "rating_normalized"
            ]
            +
            0.10
            * recommendations[
                "year_score"
            ]
        )


        n = 15


    else:

        return pd.DataFrame()


    # ==================================================
    # TRI FINAL
    # ==================================================

    recommendations = (
        recommendations.sort_values(
            "recommendation_score",
            ascending=False
        )
    )


    # IMPORTANT :
    # CE RETURN EST BIEN DANS LA FONCTION

    return recommendations.head(
        n
    )


# ==================================================
# TMDB
# ==================================================

TMDB_API_KEY = (
    st.secrets[
        "TMDB_API_KEY"
    ]
)


# ==================================================
# POSTER + RESUME FRANCAIS
# ==================================================

@st.cache_data
def get_movie_tmdb_data(
    title,
    year=None,
    imdb_id=None
):

    result = {
        "poster": None,
        "overview": None
    }


    # --------------------------------------------------
    # RECHERCHE PAR ID IMDB
    # --------------------------------------------------

    if (
        imdb_id is not None
        and pd.notna(
            imdb_id
        )
    ):

        try:

            url = (
                "https://api.themoviedb.org/"
                f"3/find/{imdb_id}"
            )


            params = {
                "api_key":
                    TMDB_API_KEY,

                "external_source":
                    "imdb_id",

                "language":
                    "fr-FR"
            }


            response = requests.get(
                url,
                params=params,
                timeout=5
            )


            response.raise_for_status()


            data = (
                response.json()
            )


            if data.get(
                "movie_results"
            ):

                movie_data = (
                    data[
                        "movie_results"
                    ][0]
                )


                poster_path = (
                    movie_data.get(
                        "poster_path"
                    )
                )


                if poster_path:

                    result[
                        "poster"
                    ] = (
                        "https://image.tmdb.org/"
                        "t/p/w500"
                        + poster_path
                    )


                overview = (
                    movie_data.get(
                        "overview"
                    )
                )


                if overview:

                    result[
                        "overview"
                    ] = overview


                return result


        except requests.RequestException:

            pass


    # --------------------------------------------------
    # RECHERCHE PAR TITRE + ANNEE
    # --------------------------------------------------

    try:

        url = (
            "https://api.themoviedb.org/"
            "3/search/movie"
        )


        params = {
            "api_key":
                TMDB_API_KEY,

            "query":
                title,

            "language":
                "fr-FR"
        }


        if pd.notna(
            year
        ):

            params[
                "year"
            ] = int(
                year
            )


        response = requests.get(
            url,
            params=params,
            timeout=5
        )


        response.raise_for_status()


        data = (
            response.json()
        )


        if data.get(
            "results"
        ):

            movie_data = (
                data[
                    "results"
                ][0]
            )


            poster_path = (
                movie_data.get(
                    "poster_path"
                )
            )


            if poster_path:

                result[
                    "poster"
                ] = (
                    "https://image.tmdb.org/"
                    "t/p/w500"
                    + poster_path
                )


            overview = (
                movie_data.get(
                    "overview"
                )
            )


            if overview:

                result[
                    "overview"
                ] = overview


    except requests.RequestException:

        pass


    return result


# ==================================================
# POURQUOI CE FILM ?
# ==================================================

def get_reasons(
    movie,
    mode
):

    reasons = []


    # --------------------------------------------------
    # MEME VIBE
    # --------------------------------------------------

    if mode == "similaire":

        if (
            movie[
                "semantic_score"
            ] >= 0.50
        ):

            reasons.append(
                "🧠 Histoire et thèmes proches"
            )


        if (
            movie[
                "weighted_genre_score"
            ] >= 0.30
        ):

            reasons.append(
                "🎭 Univers cinématographique proche"
            )


        elif (
            movie[
                "shared_genres"
            ] >= 1
        ):

            reasons.append(
                "🎭 Genre en commun"
            )


        if (
            movie[
                "year_distance"
            ] <= 5
        ):

            reasons.append(
                "🎬 Même époque"
            )


    # --------------------------------------------------
    # INCONTOURNABLES
    # --------------------------------------------------

    elif mode == "qualite":

        if (
            movie[
                "averageRating"
            ] >= 7.5
        ):

            reasons.append(
                "⭐ Très bien noté"
            )


        if (
            movie[
                "numVotes"
            ] >= 500000
        ):

            reasons.append(
                "🔥 Très populaire"
            )


        if (
            movie[
                "weighted_genre_score"
            ] >= 0.20
        ):

            reasons.append(
                "🎭 Univers compatible"
            )


    # --------------------------------------------------
    # PEPITE
    # --------------------------------------------------

    elif mode == "decouverte":

        if (
            movie[
                "averageRating"
            ] >= 7
        ):

            reasons.append(
                "⭐ Bien noté"
            )


        if (
            movie[
                "numVotes"
            ] <= 300000
        ):

            reasons.append(
                "💎 Moins connu"
            )


        if (
            movie[
                "weighted_genre_score"
            ] >= 0.20
        ):

            reasons.append(
                "🎭 Univers compatible"
            )


    if not reasons:

        reasons.append(
            "✨ Sélection FilmCity"
        )


    return reasons


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero-title">
        🎬 FilmCity
    </div>

    <div class="hero-subtitle">
        Ton prochain film commence ici 🍿
    </div>
    """,
    unsafe_allow_html=True
)


st.write(
    "Choisis un film que tu as aimé "
    "et découvre des recommandations personnalisées."
)


# ==================================================
# LISTE DES FILMS
# ==================================================

films = sorted(
    df_final[
        "primaryTitle"
    ]
    .dropna()
    .unique()
)


# ==================================================
# FILM PAR DEFAUT
# ==================================================

if (
    "selected_movie"
    not in st.session_state
):

    st.session_state[
        "selected_movie"
    ] = "Toy Story"


# ==================================================
# RECHERCHE LIVE
# ==================================================

search_movie = st.text_input(
    "🎥 Quel film as-tu aimé ?",
    value="",
    placeholder=(
        "Ex : Toy Story, Barbie, "
        "Inception..."
    )
)


if search_movie:

    matches = [
        film
        for film in films
        if search_movie.lower()
        in film.lower()
    ]


    if matches:

        st.caption(
            "🎬 Sélectionne un film :"
        )


        suggestion_columns = (
            st.columns(
                4
            )
        )


        for i, film in enumerate(
            matches[:8]
        ):

            with suggestion_columns[
                i % 4
            ]:

                if st.button(
                    film,
                    key=(
                        f"film_{i}_{film}"
                    ),
                    use_container_width=True
                ):

                    st.session_state[
                        "selected_movie"
                    ] = film


    else:

        st.warning(
            "🎬 Aucun film trouvé."
        )


selected_movie = (
    st.session_state[
        "selected_movie"
    ]
)


st.caption(
    f"✅ Film sélectionné : "
    f"{selected_movie}"
)


# ==================================================
# MODES
# ==================================================

mode_labels = {

    "🎯 Même vibe":
        "similaire",

    "⭐ Les incontournables":
        "qualite",

    "💎 Pépite à découvrir":
        "decouverte"
}


selected_label = st.radio(
    "Comment veux-tu choisir ton prochain film ?",
    list(
        mode_labels.keys()
    ),
    horizontal=True,
    index=0
)


mode = (
    mode_labels[
        selected_label
    ]
)


# ==================================================
# BADGE COLORE
# ==================================================

if mode == "similaire":

    badge_class = (
        "mode-similaire"
    )

elif mode == "qualite":

    badge_class = (
        "mode-qualite"
    )

else:

    badge_class = (
        "mode-decouverte"
    )


st.markdown(
    f"""
    <div class="
        mode-badge
        {badge_class}
    ">
        {selected_label}
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# RECOMMANDATIONS
# ==================================================

results = recommend_movie(
    selected_movie,
    mode=mode
)


# ==================================================
# AFFICHAGE
# ==================================================

if results is None:

    st.error(
        "Erreur interne dans "
        "le moteur de recommandation."
    )


elif results.empty:

    st.warning(
        "😕 Pas assez de recommandations "
        "pour ce mode."
    )


else:

    st.markdown(
        f"""
        <h2 style="
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        ">
            🍿 Parce que tu as aimé
            {selected_movie}
        </h2>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # 5 FILMS PAR LIGNE
    # --------------------------------------------------

    for start in range(
        0,
        len(
            results
        ),
        5
    ):

        columns = st.columns(
            5,
            gap="medium"
        )


        current_movies = (
            results.iloc[
                start:start + 5
            ]
        )


        for column, (
            _,
            movie
        ) in zip(
            columns,
            current_movies.iterrows()
        ):

            with column:

                # ----------------------------------
                # ID IMDB
                # ----------------------------------

                imdb_id = (
                    movie[
                        "tconst"
                    ]

                    if "tconst"
                    in movie.index

                    else None
                )


                # ----------------------------------
                # TMDB
                # ----------------------------------

                tmdb_data = (
                    get_movie_tmdb_data(
                        movie[
                            "primaryTitle"
                        ],
                        movie[
                            "startYear"
                        ],
                        imdb_id
                    )
                )


                poster = (
                    tmdb_data[
                        "poster"
                    ]
                )


                resume_fr = (
                    tmdb_data[
                        "overview"
                    ]
                )


                # ----------------------------------
                # POSTER
                # ----------------------------------

                if poster:

                    st.image(
                        poster,
                        width="stretch"
                    )


                else:

                    st.markdown(
                        """
                        <div style="
                            height: 350px;
                            background: #171D2A;
                            border-radius: 12px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            text-align: center;
                        ">
                            🎬<br>
                            Affiche indisponible
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # ----------------------------------
                # TITRE
                # ----------------------------------

                st.markdown(
                    f"""
                    <div class="film-title">
                        {
                            movie[
                                "primaryTitle"
                            ]
                        }
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ----------------------------------
                # ANNEE + NOTE
                # ----------------------------------

                st.markdown(
                    f"""
                    <div class="film-info">
                        {
                            int(
                                movie[
                                    "startYear"
                                ]
                            )
                        }
                        •
                        ⭐ {
                            movie[
                                "averageRating"
                            ]
                        }/10
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ----------------------------------
                # POURQUOI CE FILM ?
                # ----------------------------------

                with st.expander(
                    "Pourquoi ce film ?"
                ):

                    reasons = (
                        get_reasons(
                            movie,
                            mode
                        )
                    )


                    for reason in reasons:

                        st.write(
                            reason
                        )


                    st.markdown(
                        "---"
                    )


                    st.write(
                        "🎭 **Genres :**"
                    )


                    st.write(
                        movie[
                            "genres"
                        ]
                    )


                    st.write(
                        "📝 **Résumé :**"
                    )


                    if resume_fr:

                        st.write(
                            resume_fr
                        )


                    else:

                        st.write(
                            "Résumé en français "
                            "indisponible."
                        )


                    st.caption(
                        "Score interne FilmCity : "
                        f"{movie['recommendation_score'] * 100:.0f}%"
                    )