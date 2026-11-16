"""Classes GTSRB/GTSDB (43), catégories et messages d'alerte."""

NAMES = [
    "Limitation de vitesse 20 km/h", "Limitation de vitesse 30 km/h",
    "Limitation de vitesse 50 km/h", "Limitation de vitesse 60 km/h",
    "Limitation de vitesse 70 km/h", "Limitation de vitesse 80 km/h",
    "Fin de limitation à 80 km/h", "Limitation de vitesse 100 km/h",
    "Limitation de vitesse 120 km/h", "Interdiction de dépasser",
    "Interdiction de dépasser (poids lourds)", "Intersection avec priorité",
    "Route prioritaire", "Cédez le passage", "Stop", "Circulation interdite",
    "Interdit aux poids lourds", "Sens interdit", "Danger général",
    "Virage dangereux à gauche", "Virage dangereux à droite",
    "Succession de virages", "Chaussée bosselée", "Chaussée glissante",
    "Chaussée rétrécie à droite", "Travaux", "Feux tricolores",
    "Passage piétons", "Enfants", "Cyclistes", "Risque de verglas",
    "Animaux sauvages", "Fin de toutes les interdictions",
    "Direction obligatoire : à droite", "Direction obligatoire : à gauche",
    "Direction obligatoire : tout droit", "Tout droit ou à droite",
    "Tout droit ou à gauche", "Contourner par la droite",
    "Contourner par la gauche", "Sens giratoire obligatoire",
    "Fin d'interdiction de dépasser",
    "Fin d'interdiction de dépasser (poids lourds)",
]
NUM_CLASSES = len(NAMES)
assert NUM_CLASSES == 43

# Catégories officielles de GTSDB
PROHIBITORY = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
MANDATORY = set(range(33, 41))
DANGER = {11} | set(range(18, 32))
CATEGORY_KEYS = ["interdiction", "danger", "obligation", "autre"]
CATEGORY_LABELS = {
    "interdiction": "Interdiction", "danger": "Danger",
    "obligation": "Obligation", "autre": "Priorité / fin d'interdiction",
}


def category_of(cid: int) -> str:
    if cid in PROHIBITORY:
        return "interdiction"
    if cid in DANGER:
        return "danger"
    if cid in MANDATORY:
        return "obligation"
    return "autre"


# Panneaux déclenchant une alerte visuelle + sonore
ALERT_MSG = {
    14: "STOP : arrêt obligatoire",
    13: "Cédez le passage",
    17: "Sens interdit",
    25: "Travaux : ralentir",
    27: "Passage piétons : attention",
    28: "Enfants : attention",
}
ALERT_IDS = set(ALERT_MSG)
