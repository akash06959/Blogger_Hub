{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMYSUCaaJvOopfn9FvD7nkW",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/akash06959/Blogger_Hub/blob/main/cadl2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DjZamn_EBV1R",
        "outputId": "6764baaf-4d42-4034-b61f-937420a5f80b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Corpus: [\"I love this movie, it's amazing!\", 'This movie was terrible and boring.', 'The film was okay, not too bad but not great.', 'What a fantastic performance by the actor!']\n",
            "\n",
            "Bag of Words Vocabulary:\n",
            " ['actor' 'amazing' 'and' 'bad' 'boring' 'but' 'by' 'fantastic' 'film'\n",
            " 'great' 'it' 'love' 'movie' 'not' 'okay' 'performance' 'terrible' 'the'\n",
            " 'this' 'too' 'was' 'what']\n",
            "\n",
            "BoW Representation:\n",
            " [[0 1 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0]\n",
            " [0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0]\n",
            " [0 0 0 1 0 1 0 0 1 1 0 0 0 2 1 0 0 1 0 1 1 0]\n",
            " [1 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 0 1]]\n",
            "\n",
            "TF-IDF Vocabulary:\n",
            " ['actor' 'amazing' 'and' 'bad' 'boring' 'but' 'by' 'fantastic' 'film'\n",
            " 'great' 'it' 'love' 'movie' 'not' 'okay' 'performance' 'terrible' 'the'\n",
            " 'this' 'too' 'was' 'what']\n",
            "\n",
            "TF-IDF Representation:\n",
            " [[0.         0.48546061 0.         0.         0.         0.\n",
            "  0.         0.         0.         0.         0.48546061 0.48546061\n",
            "  0.38274272 0.         0.         0.         0.         0.\n",
            "  0.38274272 0.         0.         0.        ]\n",
            " [0.         0.         0.4533864  0.         0.4533864  0.\n",
            "  0.         0.         0.         0.         0.         0.\n",
            "  0.35745504 0.         0.         0.         0.4533864  0.\n",
            "  0.35745504 0.         0.35745504 0.        ]\n",
            " [0.         0.         0.         0.29823274 0.         0.29823274\n",
            "  0.         0.         0.29823274 0.29823274 0.         0.\n",
            "  0.         0.59646548 0.29823274 0.         0.         0.23513012\n",
            "  0.         0.29823274 0.23513012 0.        ]\n",
            " [0.42176478 0.         0.         0.         0.         0.\n",
            "  0.42176478 0.42176478 0.         0.         0.         0.\n",
            "  0.         0.         0.         0.42176478 0.         0.3325242\n",
            "  0.         0.         0.         0.42176478]]\n"
          ]
        }
      ],
      "source": [
        "# CADL2 – Bag of Words & TF-IDF\n",
        "from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\n",
        "\n",
        "# Sample dataset (you can replace with movie reviews, tweets, etc.)\n",
        "corpus = [\n",
        "    \"I love this movie, it's amazing!\",\n",
        "    \"This movie was terrible and boring.\",\n",
        "    \"The film was okay, not too bad but not great.\",\n",
        "    \"What a fantastic performance by the actor!\",\n",
        "]\n",
        "\n",
        "print(\"Corpus:\", corpus)\n",
        "\n",
        "# --- Bag of Words ---\n",
        "vectorizer = CountVectorizer()\n",
        "X_bow = vectorizer.fit_transform(corpus)\n",
        "print(\"\\nBag of Words Vocabulary:\\n\", vectorizer.get_feature_names_out())\n",
        "print(\"\\nBoW Representation:\\n\", X_bow.toarray())\n",
        "\n",
        "# --- TF-IDF ---\n",
        "tfidf = TfidfVectorizer()\n",
        "X_tfidf = tfidf.fit_transform(corpus)\n",
        "print(\"\\nTF-IDF Vocabulary:\\n\", tfidf.get_feature_names_out())\n",
        "print(\"\\nTF-IDF Representation:\\n\", X_tfidf.toarray())\n"
      ]
    }
  ]
}