"""
=============================================================
SYSTÈME DE NAVIGATION GPS PIÉTON - VERSION AMÉLIORÉE
=============================================================

TECHNOLOGIES UTILISÉES :
─────────────────────────────────────────────────────────────
1. PHOTON (photon.komoot.io)
   → Geocodage : convertit un nom de lieu ("Avenue Bourguiba")
     en coordonnées GPS (latitude, longitude).
   → Basé sur les données OpenStreetMap. Gratuit, sans clé API.

2. OSRM (router.project-osrm.org)
   → Calcul d'itinéraire open-source (Open Source Routing Machine).
   → Renvoie les étapes de navigation (steps) et la géométrie
     (liste de points GPS formant le chemin).
   → Profil "foot" = itinéraire piéton.

3. HAVERSINE (math pur, pas de bibliothèque)
   → Formule mathématique pour calculer la distance réelle
     en mètres entre deux points GPS sur la sphère terrestre.
   → Plus précise que le calcul euclidien simple.

4. requests
   → Bibliothèque Python standard pour faire des appels HTTP
     vers les APIs externes (Photon, OSRM).

5. time / math
   → time.sleep() : simule le temps réel de marche.
   → math : utilisé pour les calculs trigonométriques (haversine).
─────────────────────────────────────────────────────────────
"""

import requests
import time
import math

# ─────────────────────────────────────────────
# PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────

# Point de départ fixe (ici : Tunis, modifiable)
START_POS = (36.806736, 10.104995)

# Si l'utilisateur s'éloigne de plus de cette distance de la route, on recalcule
RECALCULATE_THRESHOLD = 50

# Nombre maximum de recalculs successifs pour éviter une boucle infinie
MAX_RECALCULATIONS = 5

# Pause entre chaque position simulée (en secondes) — simule la vitesse de marche
SIMULATION_STEP_PAUSE = 0.05

# Distance entre chaque point simulé (en mètres)
SIMULATION_STEP_M = 5


# ─────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────

def distance_m(a, b):
    """
    Calcule la distance réelle en mètres entre deux points GPS.
    Utilise la formule de Haversine qui tient compte de la courbure de la Terre.

    Paramètres :
        a : tuple (latitude, longitude) du point A
        b : tuple (latitude, longitude) du point B

    Retourne : distance en mètres (float)
    """
    R = 6_371_000  # Rayon moyen de la Terre en mètres

    # Conversion degrés → radians (exigé par les fonctions trigonométriques)
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    lon1, lon2 = math.radians(a[1]), math.radians(b[1])

    dlat = lat2 - lat1  # Différence de latitude
    dlon = lon2 - lon1  # Différence de longitude

    # Formule de Haversine
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(x))


def geocode_photon(place):
    """
    Convertit un nom de lieu en coordonnées GPS via l'API Photon (OpenStreetMap).

    Paramètres :
        place : chaîne de texte, ex: "Avenue Habib Bourguiba, Tunis"

    Retourne : tuple (latitude, longitude) ou None si introuvable
    """
    url = f"https://photon.komoot.io/api/?q={place}&limit=1"
    try:
        response = requests.get(url, headers={"User-Agent": "nav-app"}, timeout=10)
        response.raise_for_status()  # Lève une exception si erreur HTTP (404, 500...)
        data = response.json()

        if not data.get("features"):
            return None  # Aucun résultat trouvé

        coords = data["features"][0]["geometry"]["coordinates"]
        return (coords[1], coords[0])  # Photon renvoie [lon, lat] → on retourne (lat, lon)

    except requests.RequestException as e:
        print(f"[ERREUR RÉSEAU] Geocodage échoué : {e}")
        return None


def get_route(start, end):
    """
    Récupère un itinéraire piéton via l'API OSRM.

    Paramètres :
        start : tuple (latitude, longitude) du départ
        end   : tuple (latitude, longitude) de la destination

    Retourne :
        steps    : liste des étapes de navigation (manœuvres, distances...)
        geometry : liste de coordonnées [lon, lat] formant la route complète
        (None, None) en cas d'erreur
    """
    # OSRM attend les coordonnées au format longitude,latitude (ordre inversé !)
    url = (
        f"http://router.project-osrm.org/route/v1/foot/"
        f"{start[1]},{start[0]};{end[1]},{end[0]}"
        f"?overview=full&steps=true&geometries=geojson"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != "Ok":
            print(f"[ERREUR OSRM] Code : {data.get('code')}")
            return None, None

        steps = data["routes"][0]["legs"][0]["steps"]
        geometry = data["routes"][0]["geometry"]["coordinates"]
        return steps, geometry

    except requests.RequestException as e:
        print(f"[ERREUR RÉSEAU] Calcul de route échoué : {e}")
        return None, None


def get_instruction(step):
    """
    Génère une instruction de navigation simple et claire en français.
    L'objectif est d'avoir UNE SEULE PHRASE, courte et compréhensible.

    Paramètres :
        step : dictionnaire d'une étape OSRM

    Retourne : string, ex: "Tournez à droite dans 80 mètres"
    """
    mtype     = step["maneuver"].get("type", "")
    modifier  = step["maneuver"].get("modifier", "")
    dist      = int(step["distance"])

    # Cas d'arrivée
    if mtype == "arrive":
        return "Vous êtes arrivé à destination."

    # Départ
    if mtype == "depart":
        return f"Démarrez et continuez tout droit sur {dist} mètres."

    # Virage à gauche (sharp left, left, slight left)
    if "left" in modifier:
        return f"Tournez à gauche dans {dist} mètres."

    # Virage à droite (sharp right, right, slight right)
    if "right" in modifier:
        return f"Tournez à droite dans {dist} mètres."

    # Demi-tour
    if modifier == "uturn":
        return "Faites demi-tour."

    # Tout droit (straight, continue, etc.)
    return f"Continuez tout droit sur {dist} mètres."


def etape_active(pos, steps):
    """
    Identifie l'étape du trajet la plus proche de la position actuelle.
    Parcourt toutes les étapes et retourne celle dont le point de départ
    (maneuver location) est le plus proche de 'pos'.

    Paramètres :
        pos   : tuple (latitude, longitude) de la position actuelle
        steps : liste des étapes OSRM

    Retourne :
        best_i : index de l'étape la plus proche
        best_d : distance en mètres jusqu'au début de cette étape
    """
    best_i, best_d = 0, float("inf")

    for i, step in enumerate(steps):
        loc = step["maneuver"]["location"]  # [lon, lat] selon OSRM
        d = distance_m(pos, (loc[1], loc[0]))  # Conversion en (lat, lon)
        if d < best_d:
            best_d = d
            best_i = i

    return best_i, best_d


def fake_positions(geometry, step_m=5):
    """
    Génère une liste de positions GPS fictives simulant la marche le long du trajet.
    Interpole des points tous les 'step_m' mètres entre les coordonnées de la route.

    Paramètres :
        geometry : liste de coordonnées [lon, lat] renvoyée par OSRM
        step_m   : distance en mètres entre chaque point simulé

    Retourne : liste de tuples (latitude, longitude)
    """
    # Conversion [lon, lat] → (lat, lon)
    coords = [(c[1], c[0]) for c in geometry]
    positions = []
    acc = 0.0  # Accumulation de distance dans le segment courant

    for i in range(len(coords) - 1):
        seg_len = distance_m(coords[i], coords[i + 1])
        if seg_len == 0:
            continue  # Ignore les segments nuls (points doublons)

        while acc <= seg_len:
            t = acc / seg_len  # Paramètre d'interpolation [0, 1]
            lat = coords[i][0] + (coords[i + 1][0] - coords[i][0]) * t
            lon = coords[i][1] + (coords[i + 1][1] - coords[i][1]) * t
            positions.append((lat, lon))
            acc += step_m

        acc -= seg_len  # Reporte le surplus au segment suivant (évite les sauts)

    positions.append(coords[-1])  # S'assure que le dernier point est inclus
    return positions


# ─────────────────────────────────────────────
# BOUCLE PRINCIPALE DE NAVIGATION
# ─────────────────────────────────────────────

def naviguer(destination_nom):
    """
    Lance la navigation depuis START_POS vers une destination donnée.

    Étapes :
      1. Geocodage de la destination (nom → GPS)
      2. Calcul de l'itinéraire piéton (OSRM)
      3. Simulation de la marche point par point
      4. Affichage des instructions simplifiées
      5. Annonces anticipées avant les virages
      6. Recalcul si l'utilisateur s'éloigne trop de la route

    Paramètres :
        destination_nom : chaîne de texte, ex: "Avenue Habib Bourguiba, Tunis"
    """

    # ── ÉTAPE 1 : Geocodage ──────────────────────────────────────
    print(f"\nRecherche de : {destination_nom}")
    destination_gps = geocode_photon(destination_nom)
    if not destination_gps:
        print("Erreur : Lieu introuvable. Vérifiez le nom saisi.")
        return

    # Affichage des coordonnées de la destination
    print(f"Destination : lat={destination_gps[0]:.6f}, lon={destination_gps[1]:.6f}")

    # ── ÉTAPE 2 : Calcul de la route initiale ───────────────────
    current_pos = START_POS
    steps, geometry = get_route(current_pos, destination_gps)
    if not steps:
        print("Erreur : Impossible de calculer l'itinéraire.")
        return

    # ── ÉTAPE 3 : Génération des positions simulées ──────────────
    positions_simulees = fake_positions(geometry, step_m=SIMULATION_STEP_M)
    print(f"Points simulés générés : {len(positions_simulees)} points (tous les {SIMULATION_STEP_M}m)\n")

    # Variables d'état de la navigation
    derniere_instruction = -1  # Index de la dernière étape affichée (évite les doublons)
    nb_recalculs = 0           # Compteur de recalculs pour éviter une boucle infinie

    print("─── NAVIGATION DÉMARRÉE ───\n")

    for pos in positions_simulees:

    

        # ── A. Trouver l'étape la plus proche ───────────────────
        idx_etape, dist_waypoint = etape_active(pos, steps)

        # ── B. RECALCUL si hors de la route ─────────────────────
        if dist_waypoint > RECALCULATE_THRESHOLD:
            nb_recalculs += 1

            if nb_recalculs > MAX_RECALCULATIONS:
                print("Trop de recalculs. Vérifiez votre position.")
                break

            print(f"\nVous avez dévié de la route ({int(dist_waypoint)}m). Recalcul en cours...\n")
            steps, geometry = get_route(pos, destination_gps)

            if not steps:
                print("Recalcul impossible. Navigation arrêtée.")
                break

            positions_simulees = fake_positions(geometry, step_m=SIMULATION_STEP_M)
            derniere_instruction = -1

            for new_pos in positions_simulees:
                idx_etape, dist_waypoint = etape_active(new_pos, steps)

                if idx_etape != derniere_instruction:
                    print(get_instruction(steps[idx_etape]))
                    derniere_instruction = idx_etape

                time.sleep(SIMULATION_STEP_PAUSE)

            break

        # ── C. Affichage de l'instruction au moment du virage ───────
        if idx_etape != derniere_instruction:
            print(get_instruction(steps[idx_etape]))
            derniere_instruction = idx_etape
            nb_recalculs = 0

        time.sleep(SIMULATION_STEP_PAUSE)

    print("\n─── ARRIVÉE À DESTINATION ───\n")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Lance la navigation vers une destination à Tunis
    naviguer(input(" Choisissez une destination : "))