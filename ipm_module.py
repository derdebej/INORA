"""
ipm_module.py
=============
Module IPM (Inverse Perspective Mapping) pour estimation de distance
et décision de lisibilité OCR — sans LiDAR, caméra monoculaire uniquement.

Hypothèse : tête/caméra fixe, regardant droit devant, sol plan.

Usage rapide :
    from ipm_module import IPMEstimator, ReadabilityChecker

    ipm = IPMEstimator(cam_height=1.5, vfov=70, hfov=90, img_w=640, img_h=480)
    checker = ReadabilityChecker(ipm)

    # Dans votre boucle OCR :
    decision = checker.check(bbox_xyxy, frame, object_type="sign")
    if decision.should_ocr:
        # lancer PaddleOCR
        ...
    else:
        print(decision.voice_message)  # "Avancez de 2 pas"
"""

import math
import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("IPM")


# ─────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────

class ObjectType(Enum):
    GROUND   = "ground"    # personne, poteau, vélo  → bas bbox = pied au sol
    VERTICAL = "vertical"  # affiche, panneau, porte → surface plane verticale
    HANGING  = "hanging"   # branche, barre basse    → danger pour la tête


class ReadabilityStatus(Enum):
    READABLE     = "readable"      # OCR peut se lancer maintenant
    TOO_FAR      = "too_far"       # trop loin, guider l'utilisateur à avancer
    BAD_ANGLE    = "bad_angle"     # trop de côté, guider à se placer en face
    LOW_CONTRAST = "low_contrast"  # image trop sombre/surexposée
    UNCERTAIN    = "uncertain"     # conditions limites, essayer quand même


@dataclass
class BBox:
    """Boîte englobante en coordonnées pixel absolues."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self)  -> float: return self.x_max - self.x_min
    @property
    def height(self) -> float: return self.y_max - self.y_min
    @property
    def center_x(self) -> float: return (self.x_min + self.x_max) / 2
    @property
    def center_y(self) -> float: return (self.y_min + self.y_max) / 2
    @property
    def area(self) -> float: return self.width * self.height

    @classmethod
    def from_paddle_poly(cls, poly: list) -> "BBox":
        """
        Convertit un polygone PaddleOCR (liste de 4 points [x,y])
        en BBox axis-aligned.
        """
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2) -> "BBox":
        return cls(x1, y1, x2, y2)


@dataclass
class DistanceResult:
    """Résultat d'estimation de distance."""
    distance_m: float          # distance estimée en mètres
    method: str                # méthode utilisée
    confidence: float          # 0.0 – 1.0
    is_ground_contact: bool    # True si le bas bbox touche réellement le sol
    warning: Optional[str] = None

    @property
    def zone(self) -> str:
        d = self.distance_m
        if d < 1.0:  return "DANGER"
        if d < 2.0:  return "ALERT"
        if d < 4.0:  return "CAUTION"
        if d < 7.0:  return "INFO"
        return "FAR"

    @property
    def zone_color_bgr(self) -> tuple:
        return {
            "DANGER":  (0,   0,   220),
            "ALERT":   (0,   140, 240),
            "CAUTION": (0,   180, 100),
            "INFO":    (200, 140, 50),
            "FAR":     (150, 150, 150),
        }[self.zone]


@dataclass
class ReadabilityDecision:
    """Décision finale : lancer l'OCR ou guider l'utilisateur."""
    should_ocr: bool
    status: ReadabilityStatus
    distance_result: DistanceResult
    char_height_px: float          # hauteur estimée d'un caractère en pixels
    steps_to_walk: int             # 0 si déjà lisible
    angle_deg: float               # angle de perspective (0 = face)
    voice_message: str             # message prêt à envoyer au module TTS
    confidence: float              # 0.0 – 1.0


# ─────────────────────────────────────────────
#  Estimateur de distance IPM
# ─────────────────────────────────────────────

class IPMEstimator:
    """
    Estime la distance d'un objet à partir de sa position dans l'image,
    en utilisant la géométrie de projection inverse (IPM).

    Paramètres de calibration :
        cam_height  : hauteur de la caméra au-dessus du sol (mètres)
        vfov        : champ de vue vertical total (degrés)
        hfov        : champ de vue horizontal total (degrés)
        img_w       : largeur de l'image en pixels
        img_h       : hauteur de l'image en pixels
        horizon_y   : position de la ligne d'horizon en pixels
                      (None = calculée automatiquement depuis vfov)
        tilt_deg    : inclinaison vers le bas de la caméra (degrés, 0 = horizontal)
    """

    def __init__(
        self,
        cam_height: float = 1.5,
        vfov: float       = 70.0,
        hfov: float       = 90.0,
        img_w: int        = 640,
        img_h: int        = 480,
        horizon_y: Optional[int] = None,
        tilt_deg: float   = 10.0,
    ):
        self.cam_height = cam_height
        self.vfov       = vfov
        self.hfov       = hfov
        self.img_w      = img_w
        self.img_h      = img_h
        self.tilt_deg   = tilt_deg

        # Focale en pixels
        self.fy = img_h / (2 * math.tan(math.radians(vfov / 2)))
        self.fx = img_w / (2 * math.tan(math.radians(hfov / 2)))

        # Ligne d'horizon : centre optique ± tilt
        if horizon_y is not None:
            self.horizon_y = horizon_y
        else:
            # Avec un tilt vers le bas, l'horizon monte dans l'image
            tilt_px = self.fy * math.tan(math.radians(tilt_deg))
            self.horizon_y = int(img_h / 2 - tilt_px)

        log.debug(
            f"IPM init: cam_height={cam_height}m, vfov={vfov}°, "
            f"hfov={hfov}°, horizon_y={self.horizon_y}px, tilt={tilt_deg}°"
        )

    # ── Méthode principale ──────────────────────────────────────────────────

    def estimate_ground_distance(self, y_pixel: float) -> DistanceResult:
        """
        Distance IPM pour un point au sol (y_pixel = position verticale en px).
        Plus y_pixel est grand (bas de l'image) = plus proche.

        Retourne float('inf') si le point est au-dessus de l'horizon.
        """
        if y_pixel <= self.horizon_y:
            return DistanceResult(
                distance_m=float('inf'),
                method="ipm",
                confidence=0.0,
                is_ground_contact=False,
                warning="Point au-dessus de l'horizon — pas de sol détectable"
            )

        # Angle de dépression depuis l'axe horizontal
        delta_y   = y_pixel - self.horizon_y
        angle_rad = math.atan(delta_y / self.fy) + math.radians(self.tilt_deg)

        if angle_rad <= 0:
            return DistanceResult(float('inf'), "ipm", 0.0, False,
                                  "Angle nul ou négatif")

        distance = self.cam_height / math.tan(angle_rad)

        # Confiance : bonne entre 0.5m et 8m, dégradée en dehors
        confidence = self._distance_confidence(distance)

        return DistanceResult(
            distance_m=max(0.0, distance),
            method="ipm",
            confidence=confidence,
            is_ground_contact=True,
        )

    def estimate_vertical_surface_distance(
        self,
        bbox: BBox,
        known_real_width_m: Optional[float] = None,
    ) -> DistanceResult:
        """
        Distance à une surface verticale (affiche, panneau, porte).
        Utilise la taille apparente dans l'image (similitude de triangles).

        known_real_width_m : largeur réelle connue de l'objet en mètres.
            Si None, utilise une largeur standard de 0.8m (affiche typique).
        """
        real_w = known_real_width_m or 0.8  # affiche standard

        if bbox.width < 5:
            return DistanceResult(float('inf'), "apparent_size", 0.0, False,
                                  "Bbox trop petite pour estimer")

        distance = (real_w * self.fx) / bbox.width

        # Combiner avec IPM sur le bas de la bbox pour valider
        ipm_result = self.estimate_ground_distance(bbox.y_max)
        if ipm_result.distance_m < float('inf'):
            # Moyenne pondérée : taille apparente plus fiable pour les surfaces verticales
            combined = 0.7 * distance + 0.3 * ipm_result.distance_m
        else:
            combined = distance

        confidence = self._distance_confidence(combined) * 0.85  # légèrement moins fiable

        return DistanceResult(
            distance_m=max(0.0, combined),
            method="apparent_size",
            confidence=confidence,
            is_ground_contact=False,
        )

    def estimate_for_bbox(
        self,
        bbox: BBox,
        object_type: ObjectType = ObjectType.GROUND,
        known_real_width_m: Optional[float] = None,
    ) -> DistanceResult:
        """
        Point d'entrée unifié : choisit la méthode selon le type d'objet.
        """
        if object_type == ObjectType.GROUND:
            return self.estimate_ground_distance(bbox.y_max)

        elif object_type == ObjectType.VERTICAL:
            return self.estimate_vertical_surface_distance(bbox, known_real_width_m)

        elif object_type == ObjectType.HANGING:
            # Pour les objets suspendus : distance IPM du bas de l'objet
            result = self.estimate_ground_distance(bbox.y_max)
            result.is_ground_contact = False
            result.warning = "Objet suspendu — vérifier le dégagement pour la tête"
            return result

        return self.estimate_ground_distance(bbox.y_max)

    # ── Calibration automatique ────────────────────────────────────────────

    def calibrate_horizon(self, frame: np.ndarray) -> Optional[int]:
        """
        Détecte automatiquement la ligne d'horizon dans l'image
        via la recherche de lignes horizontales dominantes (Hough).
        Met à jour self.horizon_y si une ligne fiable est trouvée.
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return None

        horizontal_ys = []
        for line in lines:
            rho, theta = line[0]
            if abs(theta - np.pi / 2) < np.radians(15):  # ±15° de l'horizontal
                y = int(rho / math.sin(theta)) if math.sin(theta) != 0 else self.img_h // 2
                if 0 < y < self.img_h:
                    horizontal_ys.append(y)

        if horizontal_ys:
            median_y = int(np.median(horizontal_ys))
            # Accepter seulement si proche de la valeur calculée (±20%)
            if abs(median_y - self.horizon_y) < self.img_h * 0.2:
                self.horizon_y = median_y
                log.debug(f"Horizon recalibré : {self.horizon_y}px")
                return median_y

        return None

    # ── Helpers ────────────────────────────────────────────────────────────

    def _distance_confidence(self, d: float) -> float:
        """Confiance décroissante hors de la plage optimale [0.5m, 8m]."""
        if d < 0.3:   return 0.3
        if d < 0.5:   return 0.6
        if d <= 6.0:  return 0.95
        if d <= 10.0: return 0.75
        return 0.4

    def pixel_to_meters_at_distance(self, pixels: float, distance_m: float) -> float:
        """Convertit une taille en pixels en mètres réels à une distance donnée."""
        return pixels * distance_m / self.fy

    def draw_distance_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Superpose les lignes de niveau IPM sur la frame.
        Utile pour le débogage et la calibration.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        zone_distances = [
            (1.0,  (0,   0,   200), "1 m  DANGER"),
            (2.0,  (0,   120, 230), "2 m  ALERTE"),
            (4.0,  (0,   180, 80),  "4 m  VIGILANCE"),
            (7.0,  (180, 140, 40),  "7 m  INFO"),
        ]

        for dist, color, label in zone_distances:
            y = self._distance_to_y(dist)
            if 0 < y < h:
                cv2.line(out, (0, y), (w, y), color, 1)
                cv2.putText(out, label, (8, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Ligne d'horizon
        cv2.line(out, (0, self.horizon_y), (w, self.horizon_y),
                 (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(out, "horizon", (8, self.horizon_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        return out

    def _distance_to_y(self, distance_m: float) -> int:
        """Inverse de estimate_ground_distance : distance → pixel Y."""
        if distance_m <= 0:
            return self.img_h
        angle_rad = math.atan(self.cam_height / distance_m)
        effective_angle = angle_rad - math.radians(self.tilt_deg)
        delta_y = math.tan(effective_angle) * self.fy
        return int(self.horizon_y + delta_y)


# ─────────────────────────────────────────────
#  Vérificateur de lisibilité OCR
# ─────────────────────────────────────────────

class ReadabilityChecker:
    """
    Décide si une région de l'image est lisible par OCR,
    ou s'il faut guider l'utilisateur à se rapprocher/se repositionner.

    Paramètres :
        ipm                   : instance IPMEstimator
        min_char_height_px    : hauteur minimale de caractère pour OCR fiable
        target_char_height_px : hauteur visée (marge de confort)
        step_length_m         : longueur d'un pas en mètres
        avg_lines_in_sign     : nombre de lignes de texte supposées dans une affiche
        max_skew_deg          : angle de perspective maximal acceptable
    """

    def __init__(
        self,
        ipm: IPMEstimator,
        min_char_height_px: int    = 15,
        target_char_height_px: int = 22,
        step_length_m: float       = 0.75,
        avg_lines_in_sign: int     = 4,
        max_skew_deg: float        = 35.0,
    ):
        self.ipm                   = ipm
        self.min_char_height_px    = min_char_height_px
        self.target_char_height_px = target_char_height_px
        self.step_length_m         = step_length_m
        self.avg_lines_in_sign     = avg_lines_in_sign
        self.max_skew_deg          = max_skew_deg

    def check(
        self,
        bbox: BBox,
        frame: np.ndarray,
        object_type: ObjectType = ObjectType.VERTICAL,
        known_real_width_m: Optional[float] = None,
    ) -> ReadabilityDecision:
        """
        Point d'entrée principal.

        bbox        : boîte englobante de la surface à lire
        frame       : frame BGR complète (pour analyser le contraste)
        object_type : type d'objet (VERTICAL pour les affiches)
        known_real_width_m : largeur réelle si connue

        Retourne un ReadabilityDecision avec should_ocr et voice_message.
        """
        # 1. Distance
        dist_result = self.ipm.estimate_for_bbox(bbox, object_type, known_real_width_m)

        # 2. Taille des caractères estimée
        char_height = self._estimate_char_height(bbox, frame)

        # 3. Angle de perspective
        angle = self._estimate_skew_angle(bbox)

        # 4. Qualité d'image (contraste)
        contrast_ok = self._check_contrast(bbox, frame)

        # ── Décisions ──────────────────────────────────────────────────────

        # Angle trop oblique
        if angle > self.max_skew_deg:
            steps = 0
            return ReadabilityDecision(
                should_ocr=False,
                status=ReadabilityStatus.BAD_ANGLE,
                distance_result=dist_result,
                char_height_px=char_height,
                steps_to_walk=0,
                angle_deg=angle,
                voice_message="Placez-vous en face du panneau pour lire",
                confidence=0.2,
            )

        # Contraste insuffisant
        if not contrast_ok:
            return ReadabilityDecision(
                should_ocr=False,
                status=ReadabilityStatus.LOW_CONTRAST,
                distance_result=dist_result,
                char_height_px=char_height,
                steps_to_walk=0,
                angle_deg=angle,
                voice_message="Éclairage insuffisant pour lire ce panneau",
                confidence=0.3,
            )

        # Texte suffisamment grand : lancer OCR
        if char_height >= self.min_char_height_px:
            confidence = min(1.0, char_height / self.target_char_height_px)
            status = (ReadabilityStatus.READABLE
                      if char_height >= self.target_char_height_px
                      else ReadabilityStatus.UNCERTAIN)
            return ReadabilityDecision(
                should_ocr=True,
                status=status,
                distance_result=dist_result,
                char_height_px=char_height,
                steps_to_walk=0,
                angle_deg=angle,
                voice_message="Lecture en cours",
                confidence=confidence,
            )

        # Trop loin : calculer les pas nécessaires
        steps = self._steps_to_readable(char_height, dist_result.distance_m)
        msg   = self._build_voice_message(steps, dist_result)

        return ReadabilityDecision(
            should_ocr=False,
            status=ReadabilityStatus.TOO_FAR,
            distance_result=dist_result,
            char_height_px=char_height,
            steps_to_walk=steps,
            angle_deg=angle,
            voice_message=msg,
            confidence=dist_result.confidence * 0.7,
        )

    # ── Helpers internes ───────────────────────────────────────────────────

    def _estimate_char_height(self, bbox: BBox, frame: np.ndarray) -> float:
        """
        Estime la hauteur d'un caractère dans la région en pixels.
        Méthode 1 (rapide) : hauteur bbox / (nb_lignes × interlignes).
        Méthode 2 (précise) : composantes connexes sur le crop binarisé.
        Retourne la meilleure estimation disponible.
        """
        # Méthode rapide (baseline)
        quick_estimate = bbox.height / (self.avg_lines_in_sign * 1.5)

        # Méthode précise sur le crop
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(frame.shape[1], int(bbox.x_max))
        y2 = min(frame.shape[0], int(bbox.y_max))
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return quick_estimate

        try:
            gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Binarisation adaptative pour gérer les variations d'éclairage
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=15, C=8
            )
            # Composantes connexes
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            heights = []
            for i in range(1, num_labels):  # ignorer le fond (label 0)
                w_cc = stats[i, cv2.CC_STAT_WIDTH]
                h_cc = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                # Filtrer : garder seulement les composantes qui ressemblent à des lettres
                # (ratio hauteur/largeur plausible, taille minimale)
                if (5 < h_cc < bbox.height * 0.8 and
                    5 < w_cc < bbox.width * 0.7  and
                    area > 20                     and
                    0.2 < h_cc / max(w_cc, 1) < 5.0):
                    heights.append(h_cc)

            if len(heights) >= 3:
                # Médiane plus robuste que la moyenne (ignore les accents, points)
                precise = float(np.median(heights))
                log.debug(f"Char height: quick={quick_estimate:.1f}px, "
                          f"precise={precise:.1f}px ({len(heights)} composantes)")
                return precise

        except Exception as e:
            log.warning(f"Erreur analyse composantes connexes : {e}")

        return quick_estimate

    def _estimate_skew_angle(self, bbox: BBox) -> float:
        """
        Estime l'angle de perspective horizontal à partir de la bbox.
        Si la bbox est plus large qu'elle n'est haute (affiche landscape)
        et très asymétrique sur les côtés → l'utilisateur est de côté.

        Version simplifiée : compare la position horizontale du centre
        de la bbox au centre de l'image.
        """
        img_center_x = self.ipm.img_w / 2
        bbox_center_x = bbox.center_x
        offset_ratio = abs(bbox_center_x - img_center_x) / (self.ipm.img_w / 2)
        # Convertir en angle approximatif
        angle = offset_ratio * (self.ipm.hfov / 2)
        return angle

    def _check_contrast(self, bbox: BBox, frame: np.ndarray) -> bool:
        """
        Vérifie que le contraste du crop est suffisant pour l'OCR.
        Retourne False si l'image est trop sombre, surexposée, ou plate.
        """
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(frame.shape[1], int(bbox.x_max))
        y2 = min(frame.shape[0], int(bbox.y_max))
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return True  # pas d'info → ne pas bloquer

        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean    = float(np.mean(gray))
        std_dev = float(np.std(gray))

        # Trop sombre (< 30) ou surexposé (> 225) ou trop plat (std < 15)
        if mean < 30 or mean > 225 or std_dev < 15:
            log.debug(f"Contraste insuffisant : mean={mean:.1f}, std={std_dev:.1f}")
            return False
        return True

    def _steps_to_readable(self, current_char_h: float, current_dist_m: float) -> int:
        """
        Calcule le nombre de pas à faire pour atteindre la lisibilité.
        La taille du texte dans l'image est proportionnelle à 1/distance.
        """
        if current_char_h <= 0 or current_dist_m <= 0:
            return 3  # valeur par défaut sécurisée

        scale_needed   = self.target_char_height_px / max(current_char_h, 1)
        target_dist    = current_dist_m / scale_needed
        meters_to_walk = max(0.0, current_dist_m - target_dist)
        steps          = math.ceil(meters_to_walk / self.step_length_m)
        return max(1, steps)

    def _build_voice_message(self, steps: int, dist: DistanceResult) -> str:
        """Construit le message vocal adapté au contexte."""
        if dist.distance_m == float('inf'):
            return "Panneau détecté, approchez-vous pour lire"
        if steps <= 1:
            return "Avancez d'un pas pour lire ce panneau"
        if steps <= 4:
            return f"Avancez de {steps} pas pour lire ce panneau"
        return f"Panneau à environ {dist.distance_m:.0f} mètres, trop loin pour lire"


# ─────────────────────────────────────────────
#  Utilitaires de conversion PaddleOCR
# ─────────────────────────────────────────────

def paddle_poly_to_bbox(poly: list) -> BBox:
    """
    Convertit un polygone PaddleOCR dt_polys en BBox.
    poly : liste de 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    return BBox.from_paddle_poly(poly)


def get_global_text_bbox(dt_polys: list) -> Optional[BBox]:
    """
    Calcule la bbox englobante de TOUS les polygones détectés
    (utile pour traiter tout le bloc de texte comme une affiche).
    """
    if not dt_polys:
        return None
    all_x = [p[0] for poly in dt_polys for p in poly]
    all_y = [p[1] for poly in dt_polys for p in poly]
    return BBox(min(all_x), min(all_y), max(all_x), max(all_y))


def annotate_frame(frame: np.ndarray, decision: ReadabilityDecision,
                   bbox: BBox) -> np.ndarray:
    """
    Dessine la bbox et les informations IPM sur la frame.
    À utiliser pour le débogage / visualisation.
    """
    out   = frame.copy()
    color = decision.distance_result.zone_color_bgr
    x1, y1 = int(bbox.x_min), int(bbox.y_min)
    x2, y2 = int(bbox.x_max), int(bbox.y_max)

    # Boîte englobante
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    # Infos texte
    dist_str  = (f"{decision.distance_result.distance_m:.1f}m"
                 if decision.distance_result.distance_m < 999 else "?")
    char_str  = f"char~{decision.char_height_px:.0f}px"
    ocr_str   = "OCR OK" if decision.should_ocr else decision.status.value
    zone_str  = decision.distance_result.zone

    lines = [
        f"{zone_str} {dist_str}",
        char_str,
        ocr_str,
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x1 + 4, y1 - 8 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Point de contact au sol (bas de bbox)
    cv2.circle(out, (int(bbox.center_x), y2), 5, (0, 0, 255), -1)

    return out