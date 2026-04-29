"""
=============================================================
  MODULE DE RECONNAISSANCE FACIALE — LUNETTE INTELLIGENTE
  Pour personnes non-voyantes | Stack 100% gratuite & locale
=============================================================

Dépendances :
    pip install cmake
    pip install dlib
    pip install opencv-python face_recognition deepface numpy

Intégration :
    from facial_recognition_module import FacialRecognitionModule
    module = FacialRecognitionModule(tts_callback=votre_fonction_tts)

Commandes STT reconnues :
    "qui est devant moi"
    "enregistre cette personne" / "ajouter cette personne"
    "plus d'informations"
    "oublier cette personne" / "supprimer cette personne"
    "liste des personnes"
"""

import cv2
import face_recognition
import sqlite3
import numpy as np
import pickle
import os
import time
import logging
from datetime import datetime
from deepface import DeepFace

# ─── Configuration du logging ───────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Constantes ─────────────────────────────────────────────────────────────
DB_PATH         = "visages.db"
TOLERANCE       = 0.50        # Seuil de similarité (plus bas = plus strict)
CAMERA_INDEX    = 0           # Index de la caméra (0 = caméra par défaut)
CAPTURE_DELAY   = 1.0         # Délai avant capture (secondes)
NB_CAPTURES     = 3           # Nombre de captures pour moyennage


# ╔══════════════════════════════════════════════════════════════════╗
# ║              CLASSE PRINCIPALE DU MODULE                        ║
# ╚══════════════════════════════════════════════════════════════════╝

class FacialRecognitionModule:
    """
    Module de reconnaissance faciale conçu pour être intégré
    dans une lunette intelligente pour non-voyants.

    Paramètres
    ----------
    tts_callback : callable
        Fonction TTS déjà implémentée. Sera appelée avec une chaîne
        de texte à lire à voix haute.
        Exemple : tts_callback("Bonjour Ahmed")

    db_path : str, optionnel
        Chemin vers la base de données SQLite. Défaut : "visages.db"
    """

    def __init__(self, tts_callback: callable, db_path: str = DB_PATH):
        self.tts = tts_callback
        self.db_path = db_path
        self._init_db()
        self._last_frame = None
        self._last_encoding = None
        self._last_face_data = None   # données DeepFace du dernier visage
        logger.info("Module de reconnaissance faciale initialisé.")


    # ──────────────────────────────────────────────────────────────────
    # POINT D'ENTRÉE PRINCIPAL — appelé depuis main via STT
    # ──────────────────────────────────────────────────────────────────

    def handle_command(self, command: str, user_response: str = None):
        """
        Traite une commande vocale reçue depuis le module STT.

        Paramètres
        ----------
        command : str
            Texte transcrit par le STT (insensible à la casse).
        user_response : str, optionnel
            Réponse de l'utilisateur à une question posée par le module.
            Ex : nom de la personne après "Quel est son nom ?"
        """
        cmd = command.lower().strip()

        if any(k in cmd for k in ["qui est", "qui est-ce", "reconnais", "devant moi"]):
            self._reconnaitre_visage()

        elif any(k in cmd for k in ["enregistre", "ajouter", "retenir", "mémorise"]):
            self._demarrer_enregistrement()

        elif any(k in cmd for k in ["plus d'info", "plus d'information", "dis m'en plus", "en savoir plus"]):
            self._lire_infos_supplementaires()

        elif any(k in cmd for k in ["oublie", "supprimer", "effacer", "retirer"]):
            self._supprimer_personne()

        elif any(k in cmd for k in ["liste", "qui connais-tu", "personnes enregistrées"]):
            self._lister_personnes()

        elif user_response:
            # Réponse en attente après une question du module
            self._traiter_reponse_utilisateur(user_response)

        else:
            logger.warning(f"Commande non reconnue : '{command}'")


    # ──────────────────────────────────────────────────────────────────
    # 1. RECONNAISSANCE
    # ──────────────────────────────────────────────────────────────────

    def _reconnaitre_visage(self):
        """Capture un visage et tente de l'identifier."""
        self.tts("Je regarde devant vous, un instant.")

        frame, encoding = self._capturer_visage()
        if frame is None or encoding is None:
            self.tts("Je ne vois pas de visage clairement devant vous. Vérifiez l'orientation de la lunette.")
            return

        self._last_frame    = frame
        self._last_encoding = encoding

        # Cherche dans la base de données
        personne = self._chercher_dans_db(encoding)

        if personne:
            nom, relation, infos = personne["nom"], personne["relation"], personne["infos"]
            self._last_face_data = personne

            message = f"C'est {nom}"
            if relation:
                message += f", votre {relation}"
            message += "."
            self.tts(message)
            self.tts("Dites 'plus d'informations' si vous souhaitez en savoir plus.")

        else:
            # Visage inconnu → description par DeepFace
            description = self._decrire_visage_local(frame)
            self._last_face_data = None
            self.tts(description)
            self.tts("Cette personne n'est pas dans ma mémoire. "
                     "Dites 'enregistre cette personne' si vous souhaitez la retenir.")


    # ──────────────────────────────────────────────────────────────────
    # 2. DESCRIPTION LOCALE (DeepFace — 100% hors ligne)
    # ──────────────────────────────────────────────────────────────────

    def _decrire_visage_local(self, frame: np.ndarray) -> str:
        """
        Génère une description naturelle du visage via DeepFace.
        Tout se passe en local, aucune API externe.
        """
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=["age", "gender", "emotion"],
                enforce_detection=False,
                silent=True
            )
            data = result[0] if isinstance(result, list) else result

            age     = int(data.get("age", 0))
            genre   = data.get("dominant_gender", "inconnu")
            emotion = data.get("dominant_emotion", "neutre")

            # Traductions
            genre_fr = "homme" if genre.lower() == "man" else "femme"
            emotions_fr = {
                "happy":    "souriant(e)",
                "sad":      "triste",
                "neutral":  "l'air neutre",
                "angry":    "en colère",
                "surprise": "surpris(e)",
                "fear":     "apeuré(e)",
                "disgust":  "dégoûté(e)"
            }
            emotion_fr = emotions_fr.get(emotion.lower(), "l'air neutre")

            return (
                f"Devant vous, un(e) {genre_fr} d'environ {age} ans, "
                f"{emotion_fr}."
            )

        except Exception as e:
            logger.error(f"DeepFace erreur : {e}")
            return "Je détecte un visage, mais je n'arrive pas à l'analyser précisément."


    # ──────────────────────────────────────────────────────────────────
    # 3. ENREGISTREMENT
    # ──────────────────────────────────────────────────────────────────

    def _demarrer_enregistrement(self):
        """Lance le flux d'enregistrement d'un nouveau visage."""
        if self._last_frame is None or self._last_encoding is None:
            self.tts("Je dois d'abord voir la personne. Dites 'qui est devant moi' pour commencer.")
            return

        self._pending_action = "enregistrement"
        self.tts("Quel est le nom de cette personne ? Dites le nom maintenant.")


    def _finaliser_enregistrement(self, nom: str, relation: str = "", infos: str = ""):
        """Sauvegarde le visage et les informations dans la base de données."""
        try:
            encodage_blob = pickle.dumps(self._last_encoding)
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO personnes (nom, relation, infos, encodage, date_ajout) "
                "VALUES (?, ?, ?, ?, ?)",
                (nom, relation, infos, encodage_blob, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()
            self.tts(f"Parfait, j'ai retenu {nom}. Je le reconnaîtrai la prochaine fois.")
            logger.info(f"Personne enregistrée : {nom}")
        except Exception as e:
            logger.error(f"Erreur enregistrement BDD : {e}")
            self.tts("Une erreur s'est produite lors de l'enregistrement.")


    # ──────────────────────────────────────────────────────────────────
    # 4. INFOS SUPPLÉMENTAIRES
    # ──────────────────────────────────────────────────────────────────

    def _lire_infos_supplementaires(self):
        """Lit les informations supplémentaires de la dernière personne reconnue."""
        if not self._last_face_data:
            self.tts("Je n'ai pas encore reconnu de personne. Demandez d'abord qui est devant vous.")
            return

        infos    = self._last_face_data.get("infos", "")
        relation = self._last_face_data.get("relation", "")
        nom      = self._last_face_data.get("nom", "")

        if infos:
            self.tts(f"Concernant {nom} : {infos}")
        elif relation:
            self.tts(f"{nom} est votre {relation}. Aucune autre information enregistrée.")
        else:
            self.tts(f"Je n'ai pas d'informations supplémentaires sur {nom}.")


    # ──────────────────────────────────────────────────────────────────
    # 5. SUPPRESSION
    # ──────────────────────────────────────────────────────────────────

    def _supprimer_personne(self):
        """Supprime la dernière personne reconnue de la base de données."""
        if not self._last_face_data:
            self.tts("Je ne sais pas qui supprimer. Reconnaissez d'abord une personne.")
            return

        nom = self._last_face_data.get("nom", "")
        id_ = self._last_face_data.get("id")

        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("DELETE FROM personnes WHERE id = ?", (id_,))
            conn.commit()
            conn.close()
            self._last_face_data = None
            self.tts(f"J'ai oublié {nom}.")
            logger.info(f"Personne supprimée : {nom}")
        except Exception as e:
            logger.error(f"Erreur suppression : {e}")
            self.tts("Une erreur s'est produite lors de la suppression.")


    # ──────────────────────────────────────────────────────────────────
    # 6. LISTE DES PERSONNES
    # ──────────────────────────────────────────────────────────────────

    def _lister_personnes(self):
        """Lit la liste de toutes les personnes enregistrées."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT nom, relation FROM personnes ORDER BY nom")
            rows = c.fetchall()
            conn.close()

            if not rows:
                self.tts("Vous n'avez encore enregistré personne.")
                return

            noms = [f"{r[0]} ({r[1]})" if r[1] else r[0] for r in rows]
            self.tts(f"Je connais {len(noms)} personne(s) : {', '.join(noms)}.")

        except Exception as e:
            logger.error(f"Erreur lecture BDD : {e}")
            self.tts("Je n'arrive pas à lire la liste.")


    # ──────────────────────────────────────────────────────────────────
    # GESTION DES RÉPONSES UTILISATEUR (dialogue multi-tours)
    # ──────────────────────────────────────────────────────────────────

    _pending_action  = None   # Action en attente de confirmation
    _pending_nom     = None   # Nom saisi en attente de confirmation
    _pending_step    = None   # Étape courante dans le flux d'enregistrement

    def _traiter_reponse_utilisateur(self, reponse: str):
        """
        Traite les réponses vocales lors du flux d'enregistrement.

        Étapes :
            1. Recevoir le NOM
            2. Demander la RELATION (optionnel)
            3. Demander les INFOS supplémentaires (optionnel)
            4. Finaliser
        """
        reponse = reponse.strip()

        if self._pending_action == "enregistrement":

            if self._pending_step is None:
                # Étape 1 : on reçoit le nom
                self._pending_nom  = reponse
                self._pending_step = "relation"
                self.tts(
                    f"J'ai noté le nom {reponse}. "
                    "Quelle est sa relation avec vous ? "
                    "Par exemple : ami, famille, collègue. "
                    "Ou dites 'passer' pour ignorer."
                )

            elif self._pending_step == "relation":
                # Étape 2 : on reçoit la relation
                relation = "" if "passer" in reponse.lower() else reponse
                self._pending_relation = relation
                self._pending_step = "infos"
                self.tts(
                    "Voulez-vous ajouter des informations supplémentaires ? "
                    "Par exemple : son métier, son âge. "
                    "Ou dites 'passer' pour ignorer."
                )

            elif self._pending_step == "infos":
                # Étape 3 : on reçoit les infos → on finalise
                infos = "" if "passer" in reponse.lower() else reponse
                self._finaliser_enregistrement(
                    nom=self._pending_nom,
                    relation=getattr(self, "_pending_relation", ""),
                    infos=infos
                )
                # Réinitialisation du flux
                self._pending_action   = None
                self._pending_step     = None
                self._pending_nom      = None
                self._pending_relation = None

        else:
            self.tts("Je n'attendais pas de réponse. Répétez votre commande si besoin.")


    # ──────────────────────────────────────────────────────────────────
    # CAPTURE VIDÉO
    # ──────────────────────────────────────────────────────────────────

    def _capturer_visage(self):
        """
        Ouvre la caméra, effectue plusieurs captures et retourne
        la meilleure frame avec l'encodage facial moyen.

        Retourne
        --------
        (frame, encoding) ou (None, None) si aucun visage détecté.
        """
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened():
            logger.error("Impossible d'ouvrir la caméra.")
            self.tts("La caméra n'est pas disponible.")
            return None, None

        time.sleep(CAPTURE_DELAY)  # Laisse la caméra s'ajuster

        best_frame    = None
        best_encoding = None
        encodings_list = []

        for _ in range(NB_CAPTURES):
            ret, frame = cam.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            encs = face_recognition.face_encodings(rgb, locs)

            if encs:
                best_frame = frame
                encodings_list.append(encs[0])

            time.sleep(0.2)

        cam.release()

        if not encodings_list:
            return None, None

        # Moyenne des encodages pour plus de robustesse
        best_encoding = np.mean(encodings_list, axis=0)
        return best_frame, best_encoding


    # ──────────────────────────────────────────────────────────────────
    # BASE DE DONNÉES
    # ──────────────────────────────────────────────────────────────────

    def _init_db(self):
        """Crée la base de données et la table si elles n'existent pas."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS personnes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                nom        TEXT    NOT NULL,
                relation   TEXT    DEFAULT '',
                infos      TEXT    DEFAULT '',
                encodage   BLOB    NOT NULL,
                date_ajout TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"Base de données prête : {self.db_path}")


    def _chercher_dans_db(self, encoding: np.ndarray) -> dict | None:
        """
        Compare l'encodage donné avec tous ceux en base de données.

        Retourne un dict avec les infos si trouvé, sinon None.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT id, nom, relation, infos, encodage FROM personnes")
            rows = c.fetchall()
            conn.close()

            if not rows:
                return None

            known_encodings = []
            known_data      = []

            for row in rows:
                enc = pickle.loads(row[4])
                known_encodings.append(enc)
                known_data.append({
                    "id":       row[0],
                    "nom":      row[1],
                    "relation": row[2],
                    "infos":    row[3]
                })

            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx  = int(np.argmin(distances))

            if distances[best_idx] <= TOLERANCE:
                logger.info(f"Visage reconnu : {known_data[best_idx]['nom']} "
                            f"(distance={distances[best_idx]:.3f})")
                return known_data[best_idx]

            return None

        except Exception as e:
            logger.error(f"Erreur recherche BDD : {e}")
            return None


    def ajouter_personne_manuellement(self, nom: str, relation: str = "",
                                       infos: str = "", frame=None):
        """
        API publique pour ajouter une personne directement
        (utile pour les tests ou l'administration).
        """
        if frame is None:
            frame, encoding = self._capturer_visage()
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            encoding = encs[0] if encs else None

        if encoding is None:
            self.tts("Aucun visage détecté pour l'enregistrement.")
            return False

        self._last_encoding = encoding
        self._finaliser_enregistrement(nom, relation, infos)
        return True


# ╔══════════════════════════════════════════════════════════════════╗
# ║          EXEMPLE D'INTÉGRATION DANS VOTRE main()               ║
# ╚══════════════════════════════════════════════════════════════════╝

def exemple_integration():
    """
    Exemple minimal montrant comment intégrer ce module
    dans votre fichier main.py existant.
    """

    # --- Votre fonction TTS (déjà implémentée) ---
    def mon_tts(texte: str):
        # Remplacez par votre module TTS réel
        print(f"[TTS] ▶ {texte}")

    # --- Initialisation du module ---
    module_facial = FacialRecognitionModule(tts_callback=mon_tts)

    # --- Simulation d'une boucle principale STT ---
    print("\n=== Lunette Intelligente — Module Facial ===")
    print("Commandes disponibles :")
    print("  'qui est devant moi'")
    print("  'enregistre cette personne'")
    print("  'plus d'informations'")
    print("  'liste des personnes'")
    print("  'quitter'\n")

    en_attente_reponse = False

    while True:
        # Simuler la réception d'une commande STT
        commande = input("Commande STT > ").strip()

        if commande.lower() == "quitter":
            print("Au revoir.")
            break

        if en_attente_reponse:
            # Le module attend une réponse (ex : saisie du nom)
            module_facial.handle_command("", user_response=commande)
            # Vérifier si on est encore en attente
            en_attente_reponse = (module_facial._pending_action is not None)
        else:
            module_facial.handle_command(commande)
            # Détecter si le module attend une réponse
            en_attente_reponse = (module_facial._pending_action is not None)


if __name__ == "__main__":
    exemple_integration()