"""Rule-based multilingual intent detection with typo tolerance.

Several intents can match one message (multiple requests). Confidence rises with the
number of distinct patterns that fire. High-risk intents (account compromise, duplicate
payment, legal threat) are matched in English, Hindi (Devanagari + romanised), Spanish, French.
"""
import difflib
import re
from dataclasses import dataclass

from ..core.text import fold

_PATTERNS: dict[str, list[str]] = {
    "account_compromise": [
        r"\b(?:hacked|compromised|phished|account (?:was |got |has been )?(?:breached|stolen|taken over|hacked)|stolen account)\b",
        r"\bunauthori[sz]ed (?:login|access|transaction|purchase|activity|charge)s?\b",
        r"\bsomeone (?:else )?(?:logged|accessed|is using|has access|got into|broke into)\b",
        r"\b(?:logged in|login) from (?:an? )?(?:unknown|unfamiliar|strange)\b",
        r"\bpassword (?:was |got )?changed (?:without|by someone)\b",
        r"\bhack ho\b|\baccount hack\b|\bkisi ne (?:mera )?account\b|हैक|खाता.*हैक",
        r"\bhackearon\b|\bcuenta (?:fue |ha sido )?(?:hackeada|comprometida|robada)\b|\bacceso no autorizado\b|\balguien (?:entro|accedio|esta usando)\b",
        r"\bcompte (?:a ete |a été )?(?:pirate|hacke|compromis|vole)\b|\bacces non autorise\b|\bquelqu.un (?:s.est connecte|a acces|utilise)\b",
    ],
    "duplicate_payment": [
        r"\b(?:charged|billed|debited|deducted|paid)\b[^.\n]{0,25}\b(?:twice|two times|2 times|double|again)\b",
        r"\b(?:twice|double|duplicate)\b[^.\n]{0,20}\b(?:charge[ds]?|payment|billing|deduction|debit|transaction)\b",
        r"\bdouble[- ]?charg\w*|\bduplicate (?:charge|payment|transaction|billing)\b|\bpaid twice\b|\btwo (?:separate )?(?:charges|payments)\b",
        r"\bdo baar\b[^.\n]{0,20}\b(?:paise|paisa|payment|kat|cut)\b|\b(?:paise|paisa)\b[^.\n]{0,15}\bdo baar\b|\bdobara (?:paise|paisa)\b|दो बार",
        r"\bcobr\w+ dos veces\b|\bdoble cobro\b|\bcobro doble\b|\bpago duplicado\b|\bpague dos veces\b",
        r"\bdebit\w* deux fois\b|\bpaye deux fois\b|\bdouble debit\b|\bpaiement en double\b|\bfacture\w* deux fois\b",
    ],
    "legal_threat": [
        r"\b(?:lawyer|attorney|solicitor|legal action|legal notice|legal team|legal proceedings|lawsuit|suing)\b|\bsue\b",
        r"\btake (?:you|this|the company|it) to court\b|\bconsumer (?:court|forum|protection)\b|\bfile a (?:case|complaint) (?:against|in court)\b",
        r"\bvakeel\b|\bkanooni\b|\bcourt mein\b|कानूनी|वकील|उपभोक्ता",
        r"\babogado\b|\bdemandar\b|\bdemanda legal\b|\btribunal\b|\baccion legal\b",
        r"\bavocat\b|\baction en justice\b|\bpoursuivre en justice\b|\btribunal\b",
    ],
    "refund": [
        r"\b(?:refund|refunded|money back|reimburse\w*|return my money)\b",
        r"\bpaise wapas\b|\brefund chahiye\b|\bwapas chahiye\b|रिफंड|पैसे वापस",
        r"\breembolso\b|\bdevuelv\w+ (?:mi )?dinero\b",
        r"\bremboursement\b|\brembourser\b",
    ],
    "order_status": [
        r"\bwhere is my (?:order|package|parcel|delivery)\b|\btrack\w*\b|\border status\b|\bdelivery (?:status|date|delay\w*|late)\b",
        r"\b(?:not|hasn.t|haven.t|never)\b[^.\n]{0,15}\b(?:arrived|delivered|received|shipped)\b",
        r"\border nahi (?:aaya|aya|pahuncha|mila)\b|\bkab (?:aayega|milega|ayega)\b|नहीं (?:आया|पहुंचा|मिला)",
        r"\bdonde esta mi pedido\b|\bno (?:ha )?llegado\b|\bmi pedido no\b|\bno (?:he )?recibido\b",
        r"\bou est ma commande\b|\bpas (?:encore )?(?:recu|arrive|livre)\b|\bcommande n.est pas\b",
    ],
    "technical_issue": [
        r"\b(?:not working|doesn.t work|does not work|stopped working|broken|crash\w*|defective|faulty|damaged|no power|blue screen|won.t (?:turn on|start|connect|charge))\b",
        r"\bkeeps? (?:disconnecting|crashing|restarting|freezing)\b|\berror\b|\bglitch\w*\b",
        r"\bkaam nahi kar\w*\b|\bchal nahi\b|\bkharab\b|\bband ho gaya\b|काम नहीं|खराब",
        r"\bno funciona\b|\bdejo de funcionar\b|\broto\b|\bdanado\b",
        r"\bne marche pas\b|\bne fonctionne pas\b|\ben panne\b|\bcasse\b|\bdefectueux\b|\berreur\b",
    ],
    "policy_question": [
        r"\b(?:return|refund|warranty|exchange|cancellation|shipping|privacy|replacement) (?:policy|period|window|terms|conditions)\b",
        r"\bhow (?:long|many days)\b|\bwhat is (?:your|the) (?:policy|warranty|return)\b|\bcan i (?:return|exchange)\b|\bis (?:it|this) covered\b|\bwarranty\b",
        r"\bpolicy kya\b|\bkitne din\b|\bwapsi ki policy\b",
        r"\bpolitica de (?:devolucion|reembolso|garantia)\b|\bcuantos dias\b|\bgarantia\b",
        r"\bpolitique de (?:retour|remboursement)\b|\bgarantie\b|\bcombien de jours\b",
    ],
    "cancel_order": [
        r"\bcancel (?:my |the )?(?:order|subscription|purchase)\b|\border cancel\b|\bcancelar (?:mi )?pedido\b|\bannuler (?:ma )?commande\b",
    ],
    "human_agent": [
        r"\b(?:talk|speak|connect me) to (?:a |an |the )?(?:human|agent|person|manager|representative|supervisor)\b|\breal person\b|\binsaan se\b|\bagente humano\b|\bparler a un (?:humain|agent|conseiller)\b",
    ],
    "resolved": [
        r"\b(?:it|that|this|everything)(?:.s| is| has been| was)? (?:fixed|resolved|working now|sorted)\b|\bproblem (?:is )?(?:solved|resolved)\b|\bworks now\b|\bworking now\b|\bissue (?:is )?resolved\b|\bye(?:s)? it worked\b",
    ],
    "greeting": [r"^(?:hi|hello|hey|namaste|hola|bonjour|salut|good (?:morning|afternoon|evening))\b"],
    "thanks": [r"\bthank(?:s| you)\b|\bshukriya\b|\bdhanyavad\b|\bgracias\b|\bmerci\b|धन्यवाद|शुक्रिया"],
    "complaint": [
        r"\b(?:not happy|disappointed|unacceptable|terrible service|worst|complain\w*|ridiculous|fed up|useless|pathetic)\b",
        r"\bbahut kharab\b|\bmuy mal\b|\btres mauvais\b",
    ],
}
# ---------------------------------------------------------------------------------------------------------------
# HIGH-RISK intents, broadened by CONCEPT. Layer 1: extra regexes. Layer 2: order-independent concept co-occurrence.
# Exclusions (how-to questions, hypotheticals, third parties, relatives, "twice as much") apply to BOTH layers.
# ---------------------------------------------------------------------------------------------------------------
_ACC_OBJ = r"(?:log-?ins?|sign-?ins?|transactions?|charges?|purchases?|orders?|access|devices?|activity|activities|payments?|withdrawals?|transfers?)"
_PATTERNS["account_compromise"] += [
    rf"\b(?:unauthori[sz]ed|unrecogni[sz]ed|unknown|suspicious|fraudulent|illegitimate|strange|unfamiliar)\b[^.\n]{{0,25}}\b{_ACC_OBJ}\b",
    r"\b(?:someone|somebody|some one|an unknown person|a stranger|a third party|hackers?|scammers?|criminals?|thieves|a thief)\b[^.\n]{0,35}"
    r"\b(?:used|using|logged|accessed|access|took over|taken over|got into|broke into|has access|changed|stole|stolen|made|placed|opened|"
    r"entered|hacked|hijacked|compromised|impersonat\w*|pretending|phished)\b",
    r"\b(?:account|profile|login|password|email|card)\b[^.\n]{0,25}\b(?:hack\w*|hijack\w*|breach\w*|compromis\w*|taken over|phish\w*)\b",
    r"\b(?:hack\w*|hijack\w*|breach\w*|compromis\w*|phish\w*)\b[^.\n]{0,25}\b(?:account|profile|login|password|email|card)\b",
    r"\baccount takeover\b|\bwas hacked\b|\bgot hacked\b|\bbeen hacked\b",
    r"\b(?:log-?in|sign-?in|logged in|signed in)\b[^.\n]{0,30}\b(?:unknown|unfamiliar|strange|new|different|another)\b[^.\n]{0,15}\b(?:device|location|country|place|ip|browser)\b",
    r"\b(?:device|location|country|ip)\b[^.\n]{0,15}\b(?:i )?(?:do not|don.t|dont|did not|didn.t) (?:recogni[sz]e|know|use|own)\b",
    r"\b(?:password|e-?mail|phone number|address|recovery)\b[^.\n]{0,30}\b(?:changed|reset|updated|altered|modified)\b[^.\n]{0,40}"
    r"\b(?:without|didn.t|did not|never|not by me|wasn.t me)\b",
    r"\b(?:changed|reset|updated|altered)\b[^.\n]{0,25}\b(?:password|e-?mail)\b[^.\n]{0,40}\b(?:without|didn.t|did not|never|not by me|wasn.t me)\b",
    r"\b(?:otp|one[- ]time (?:password|code)|verification code|reset (?:email|link|code|request)|password reset|security code)\b[^.\n]{0,45}"
    r"\b(?:didn.t|did not|never|haven.t|wasn.t)\b[^.\n]{0,15}\b(?:request|ask|order|initiate|trigger|send|make)\w*",
    rf"\b{_ACC_OBJ}\b[^.\n]{{0,30}}\b(?:i|we)\s+(?:didn.t|did not|never|haven.t|have not)\s+(?:make|made|place|placed|authori[sz]e|authori[sz]ed|approve|approved|do|did|initiate|initiated|perform|performed|request|requested)\b",
    rf"\b(?:i|we)\s+(?:didn.t|did not|never)\s+(?:make|made|place|placed|authori[sz]e|approve|approved|initiate|perform)\b[^.\n]{{0,30}}\b{_ACC_OBJ}\b",
    r"\bnot (?:made|placed|authori[sz]ed|requested|done) by me\b|\bwasn.t me\b|\bit was not me\b",
    r"\b(?:card|identity|credentials?|password|account|details|data)\b[^.\n]{0,20}\b(?:stolen|was stolen|got stolen|leaked|exposed|skimmed|cloned)\b",
    r"\b(?:stolen|leaked|cloned|skimmed)\b[^.\n]{0,15}\b(?:card|identity|credentials?|password|account)\b",
    r"\blocked out\b[^.\n]{0,60}\b(?:someone|hack\w*|suspicious|took over|taken over|stranger)\b",
    r"\bfraud(?:ulent)?\b[^.\n]{0,20}\b(?:charge|transaction|purchase|activity|payment|card|login|account)s?\b|"
    r"\b(?:charge|transaction|purchase|activity|payment|card|login|account)s?\b[^.\n]{0,15}\bfraud(?:ulent)?\b",
    # Spanish
    r"\balguien\b[^.\n]{0,30}\b(?:accedio|entro|uso|usa|esta usando|ingreso|inicio sesion|cambio|robo|hackeo|compro|hizo)\b",
    r"\bsin mi (?:permiso|autorizacion|consentimiento|conocimiento)\b",
    r"\b(?:cargos?|compras?|transacciones?|pagos?|movimientos?)\b[^.\n]{0,25}\b(?:no reconozco|no hice|no autorice|no realice|desconocid\w*|no son mios)\b|\bno reconozco\b",
    r"\brobaron mi (?:cuenta|tarjeta|identidad|contrasena)\b|\btarjeta (?:robada|clonada)\b|\bcuenta\b[^.\n]{0,15}\b(?:vulnerada|violada)\b",
    # French
    r"\bquelqu.un\b[^.\n]{0,30}\b(?:s.est connecte|a acces|utilise|a utilise|se sert|est entre|a pirate|a change|a modifie|a vole|a pris)\b",
    r"\b(?:achats?|operations?|transactions?|paiements?|commandes?|connexions?)\b[^.\n]{0,30}\b(?:que je n.ai pas|non autorise\w*|inconnu\w*|que je ne reconnais pas)\b",
    r"\bje ne reconnais pas\b|\bn.ai pas (?:fait|autorise|effectue|passe|demande)\b|\bsans mon (?:accord|autorisation|consentement|aval)\b",
    r"\bcarte (?:volee|piratee|clonee)\b|\busurpation d.identite\b",
    # Hindi
    r"\bkisi (?:aur |dusre )?ne\b[^.\n]{0,40}\b(?:login|account|order|use|access|password|badla|badal|hack|kiya)\b",
    r"\bmaine nahi (?:kiya|kiye|kia|ki|kari|manga|mangaya)\b|\bjo maine nahi\b|\bbina (?:meri )?(?:ijazat|permission|anumati)\b|\bmeri (?:ijazat|permission) ke bina\b",
    "किसी (?:और |दूसरे )?ने[^.\\n]{0,30}(?:लॉगिन|लोगिन|अकाउंट|खाते|खाता|पासवर्ड|ऑर्डर|इस्तेमाल)",
    "मैंने नहीं (?:किया|किए|की)", "बिना (?:मेरी )?(?:अनुमति|इजाज़त|इजाजत)",
    # a third party + something that is MINE; passive voice; "I never changed it"; alibi; "without my knowledge"
    r"\b(?:someone|somebody|some one|stranger|third party|hackers?|scammers?|unknown person|thief|criminals?)\b[^.\n]{0,60}\b(?:my|the)\s+(?:account|login|log-?in|card|password|profile|e-?mail|details|identity)\b",
    r"\b(?:account|profile|login)\b[^.\n]{0,20}\b(?:was|has been|got|were|been)\s+(?:accessed|used|logged into|entered|opened|taken|changed|compromised|tampered)\b",
    r"\b(?:i|we)\s+(?:never|didn.t|did not|have not|haven.t)\s+(?:change|changed|update|updated|modify|modified|edit|edited|reset|authori[sz]e|authori[sz]ed|approve|approved|place|placed|request|requested)\b[^.\n]{0,20}\b(?:it|that|them|this|these|those)\b",
    r"\b(?:purchases?|orders?|transactions?|charges?|logins?|activity)\b[^.\n]{0,40}\bi was (?:asleep|sleeping|away|abroad|offline|on holiday|at work|on a (?:flight|plane|trip))\b",
    r"\bwithout my (?:knowledge|permission|consent|authori[sz]ation|approval|awareness)\b",
    r"\bon a (?:pirate|hacke|vole|usurpe)\b[^.\n]{0,20}\b(?:compte|carte|identite|mot de passe)\b|\b(?:ils|elle|il) (?:a|ont) (?:pirate|hacke|vole)\b",
    r"\b(?:me|mi|mis)\b[^.\n]{0,15}\b(?:cambiaron|robaron|hackearon)\b|\balguien cambi\w* mi (?:contrasena|clave|correo)\b|\bcompras? en mi cuenta que (?:yo )?no\b",
    r"\bkisi ne\b[^.\n]{0,25}\b(?:mera|meri)\b[^.\n]{0,15}\b(?:password|account|card|login)\b|\bmaine nahi\b[^.\n]{0,15}\b(?:kiye|kiya)\b",
    r"\b(?:suspect|think|believe|feel|fear|sure|certain|convinced)\b[^.\n]{0,25}\b(?:hack\w*|breach\w*|compromis\w*|fraud|stolen|taken over|hijack\w*|phish\w*)\b",
    r"\b(?:isn.t|is not|aren.t|are not|wasn.t|was not|weren.t|were not)\s+(?:mine|my own|from me|made by me|placed by me|done by me)\b",
    rf"\b(?:don.t|do not|didn.t|did not|can.t|cannot|couldn.t)\s+(?:recogni[sz]e|remember|know about|know of)\b[^.\n]{{0,30}}\b{_ACC_OBJ}\b",
    rf"\b{_ACC_OBJ}\b[^.\n]{{0,30}}\b(?:i\s+)?(?:don.t|do not|didn.t|did not|can.t|cannot)\s+(?:recogni[sz]e|remember|know about|know of)\b",
    r"\bi\s+(?:wasn.t|was not|am not)\s+the\s+(?:one|person)\s+(?:who|that)\b",
    r"\b(?:e-?mail|password|phone number|recovery(?: e-?mail)?)\b[^.\n]{0,30}\b(?:had been|has been|was|were)\s+(?:replaced|swapped|overwritten|removed)\b",
]

_PATTERNS["duplicate_payment"] += [
    r"\b(?:charg\w*|bill\w*|debit\w*|deduct\w*|paid|taken|withdr\w*|took|processed|collected)\b[^.\n]{0,40}"
    r"\b(?:twice|two times|2 times|2x|double|doubled|second time|more than once|multiple times|a second time)\b",
    r"\b(?:charg\w*|bill\w*|debit\w*|deduct\w*)\b[^.\n]{0,30}\bagain\b",
    r"\b(?:twice|two times|2 times|double|duplicate|duplicated|second|two identical|two separate|two same|identical|repeated)\b[^.\n]{0,25}"
    r"\b(?:charges?|payments?|deductions?|debits?|billing|transactions?|withdrawals?)\b",
    r"\b(?:payment|charge|transaction|amount|money|purchase|deduction|debit|withdrawal)s?\b[^.\n]{0,30}"
    r"\b(?:twice|two times|2 times|doubled|duplicated|processed (?:twice|two times)|more than once|multiple times)\b",
    r"\bwent through (?:twice|two times)\b",
    r"\b(?:cobr\w*|carg\w*|pag\w*|descont\w*|debit\w*|retir\w*)\b[^.\n]{0,30}\b(?:dos veces|doble|duplicad\w*|2 veces|segunda vez|otra vez)\b",
    r"\b(?:cargo|pago|cobro|descuento|debito)s?\s+(?:duplicad\w*|doble)\b|\bdoble (?:cargo|cobro|pago|descuento)\b",
    r"\b(?:debit\w*|pay\w*|factur\w*|preleve\w*|prelev\w*)\b[^.\n]{0,30}\b(?:deux fois|en double|2 fois|une seconde fois)\b",
    r"\bdouble (?:debit|prelevement|paiement|facturation)\b|\bpaiement en double\b|\bprelevement en double\b|\bdeux (?:prelevements|paiements|debits)\b",
    r"\b(?:do baar|dobara|dubara|doosri baar|2 baar|do bar)\b[^.\n]{0,25}\b(?:paise|paisa|payment|kat|cut|deduct|charge|pay|bhugtan|debit)\w*",
    r"\b(?:paise|paisa|payment|bhugtan|amount)\b[^.\n]{0,25}\b(?:do baar|dobara|dubara|doosri baar|2 baar|do bar)\b",
    "(?:दो बार|दुबारा)[^.\\n]{0,25}(?:पैसे|भुगतान|कट|चार्ज)", "(?:पैसे|भुगतान)[^.\\n]{0,25}(?:दो बार|दुबारा)",
    r"\b(?:two|2)\s+(?:payments?|charges?|transactions?|deductions?|debits?|withdrawals?|bills?)\b[^.\n]{0,30}\b(?:same|one|single|1|identical)\b",
    r"\b(?:payment|money|amount|charge|fee)\b[^.\n]{0,20}\b(?:taken|deducted|charged|debited|collected|withdrawn)\s+again\b",
    r"\balready paid\b[^.\n]{0,40}\bagain\b|\bagain\b[^.\n]{0,40}\balready paid\b|\b(?:taken|collected|withdrawn)\b[^.\n]{0,20}\bagain\b",
    r"\b(?:left|leaves?|came out of|came from) (?:my|the) (?:account|bank|card|wallet)\b[^.\n]{0,20}\b(?:twice|two times)\b",
]
_DUP_EXCLUDE = re.compile(
    r"\b(?:installments?|a year|per year|each year|annual\w*|monthly|per month|a month|can i pay|could i pay|may i pay|how (?:do|can) i pay|"
    r"twice as|twice the|double the (?:price|cost|amount|quantity|size)|twice (?:more|that))\b")

_PATTERNS["legal_threat"] += [
    r"\b(?:sue|suing|sued|lawsuit|litigation|class action|small claims|cease and desist|defamation)\b",
    r"\blegal (?:action|steps|notice|proceedings|remedy|remedies|recourse|team|counsel|case|trouble|complaint|claim)\b",
    r"\b(?:take|taking|took|bring|bringing|drag|dragging|going|go|see)\b[^.\n]{0,25}\b(?:to|before|into|in) (?:the )?(?:court|a judge|tribunal|magistrate)\b|"
    r"\bsee you in court\b|\bcourt of law\b|\bfile (?:a )?(?:suit|case|petition|fir)\b",
    r"\b(?:consumer|customer)\s+(?:court|forum|commission|protection|rights|affairs|council|authority)\b",
    r"\b(?:regulator|ombudsman|trading standards|attorney general|financial authority|data protection (?:authority|commissioner))\b",
    r"\breport(?:ing)?\b[^.\n]{0,25}\b(?:to|with)\b[^.\n]{0,20}\b(?:authorit\w+|police|regulator|ombudsman)\b",
    r"\b(?:police|fir|cyber ?crime)\b[^.\n]{0,20}\b(?:complaint|report|case)\b|\b(?:complaint|report|case)\b[^.\n]{0,15}\b(?:police|cyber ?crime)\b|\bgo(?:ing)? to the police\b",
    r"\b(?:contact(?:ing)?|call(?:ing)?|consult(?:ing)?|hir(?:e|ing)|speak(?:ing)? (?:to|with)|talk(?:ing)? (?:to|with)|involv(?:e|ing)|instruct(?:ing)?|"
    r"retain(?:ing)?|get(?:ting)?|see(?:ing)?|meet(?:ing)?|inform(?:ing)?|notify(?:ing)?|go(?:ing)? to)\b[^.\n]{0,15}"
    r"\b(?:lawyers?|attorneys?|solicitors?|advocates?|legal (?:team|counsel|department|advisor|adviser))\b",
    r"\b(?:my|our)\s+(?:lawyers?|attorneys?|solicitors?|advocates?|legal (?:team|counsel|advisor|adviser))\b(?!\s+(?:friend|brother|sister|cousin|uncle|aunt|father|mother|dad|mom|colleague|neighbou?r|buddy|husband|wife|son|daughter))",
    r"\b(?:demandar\w*|denuncia\w*|querella|juzgado|acciones? legal(?:es)?|medidas legales|via judicial|profeco|indecopi|sernac)\b",
    r"\b(?:poursuite\w*|poursuivre|saisir (?:le )?(?:tribunal|justice)|mise en demeure|dgccrf|huissier)\b|"
    r"\b(?:deposer|depose|deposerai|porter|porterai|porte|engager)\b[^.\n]{0,15}\bplainte\b",
    r"\bvakil\b|\bcourt (?:mein|me|jaunga|jaungi)\b|\bcase (?:karunga|karungi|kar dunga|dalunga|file karunga)\b|\bpolice (?:mein|me|complaint)\b|\bfir (?:karunga|darj|likhwa)\w*",
    "कानूनी", "वकील", "अदालत", "कोर्ट", "मुकदमा", "उपभोक्ता (?:फोरम|अदालत|आयोग)", "पुलिस (?:में|शिकायत)",
    r"\b(?:presentar|presentare|interponer|poner|iniciar|hacer)\w*\b[^.\n]{0,12}\bdemanda\b",
]
# a bare mention of a lawyer is too weak: replace the old catch-all (so "my brother is a lawyer" is not a threat)
_PATTERNS["legal_threat"][0] = (r"\b(?:attorney|solicitor)\b[^.\n]{0,25}\b(?:will|going|contact|call|letter|notice|about this|handle)\b|\bsue\b|\blegal action\b|"
                                r"\blegal notice\b|\blegal team\b|\blegal proceedings\b|\blawsuit\b|\bsuing\b")

_DEV = re.compile(r"[\u0900-\u097f]")


def _compile(pattern: str) -> re.Pattern:
    # Devanagari patterns must be folded exactly like the text they are matched against (nukta forms)
    return re.compile(fold(pattern) if _DEV.search(pattern) else pattern, re.I)


_COMPILED = {k: [_compile(p) for p in v] for k, v in _PATTERNS.items()}

# ---------------------------------------------------------------------------------------------------------------
# CONCEPT layer: order-independent co-occurrence inside one sentence (or two adjacent ones).
#   account compromise = something of mine + an "unauthorised / unknown / someone else" marker
#   duplicate payment  = a money word + a "twice / double / duplicate" marker
#   legal threat       = a legal word + a "will / going to / seeking" marker
# ---------------------------------------------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")
_ADV = r"(?:definitely |certainly |absolutely |really |clearly |obviously |actually |honestly )?"


def _c(pattern: str) -> re.Pattern:
    return re.compile(fold(pattern) if _DEV.search(pattern) else pattern, re.I)


_ACC_STRONG = _c(r"\b(?:account|accounts|login|logins|log-?in|sign-?in|password|passcode|pin|profile|card|cards|wallet|credentials?|identity|e-?mail|"
                 r"cuenta|contrasena|clave|tarjeta|correo|identidad|compte|carte|mot de passe|identite|khata)\b|खाता|खाते|अकाउंट|पासवर्ड|कार्ड")
_ACC_ACTIVITY = _c(r"\b(?:orders?|purchases?|transactions?|charges?|history|activity|activities|pedidos?|compras?|cargos?|historial|commandes?|achats?|historique|"
                   r"operations?|logins?)\b|ऑर्डर")
_UNAUTH = _c(r"\b(?:unknown|unrecogni[sz]ed|unfamiliar|strange|suspicious|fraudulent|fake|scam\w*|hack\w*|phish\w*|hijack\w*|breach\w*|compromis\w*|"
             r"stolen|stole|steal\w*|impersonat\w*|someone|somebody|strangers?|random person|third party|thief|unauthori[sz]ed|tampered|swapped|"
             r"another person|other person|a different person|someone else|somebody else|other people|unknown people|"
             r"not mine|know nothing about|never been|never visited|never used|never seen|nothing to do with|without my|wasn.t me|not me|not by me|"
             r"while i was (?:asleep|sleeping|away|abroad|travell?ing|on a (?:flight|plane|trip|holiday|vacation)|at work|offline))\b|"
             rf"\b(?:i|we)\s+{_ADV}(?:never|didn.t|did not|haven.t|have not|wasn.t|was not)\s+(?:make|made|place|placed|order|ordered|authori[sz]e|authori[sz]ed|"
             r"approve|approved|request|requested|change|changed|update|updated|do|did|initiate|buy|bought|purchase|purchased|create|created|open|opened|"
             r"sign|signed|log|logged|use|used)\b|"
             r"\b(?:desconocid\w*|alguien|no son mios|no hice|no autorice|sin mi|sin que yo|otra persona|terceros|falso\w*|robaron|hackearon|sospechos\w*|no reconozco|"
             r"inconnu\w*|quelqu.un|quelqu.un d.autre|une autre personne|un tiers|a mon insu|jamais passe|pas fait|n.ai pas (?:fait|passe|autorise)|"
             r"ce n.est pas moi|pas moi|suspects?|sans mon|sans que je|frauduleu\w*|vole|pirate|hacke|kisi ne|maine (?:kabhi )?nahi|mene nahi|anjaan|bina meri|hack ho)\b|"
             r"\b(?:koi aur|koi dusra|koi anjaan)\b|किसी ने|कोई और|मैंने नहीं|हैक|अनजान|बिना")
# for ACTIVITY nouns a plain "someone" is not enough (a parcel delivered to someone else is not a hacked account)
_UNAUTH_STRICT = _c(r"\b(?:unknown|unrecogni[sz]ed|unfamiliar|suspicious|suspects?|fraudulent|fake|know nothing about|not mine|never been|never visited|"
                    r"while i was (?:asleep|sleeping|away|abroad|travell?ing|on a (?:flight|plane|trip|holiday|vacation)|at work|offline)|"
                    r"under my name|without my|wasn.t me|not me|ce n.est pas moi|unauthori[sz]ed|stolen|hack\w*|no reconozco|no son mios|falso\w*|inconnu\w*|"
                    r"frauduleu\w*|anjaan)\b|"
                    rf"\b(?:i|we)\s+{_ADV}(?:never|didn.t|did not|haven.t|have not)\s+(?:make|made|place|placed|order|ordered|authori[sz]e|authori[sz]ed|approve|approved|"
                    r"request|requested|initiate|buy|bought|purchase|purchased)\b|\bque je n.ai (?:jamais|pas) (?:passe|fait)\w*|\bque (?:yo )?no (?:hice|autorice)\b|"
                    r"\bjo (?:maine|mene) (?:kabhi )?nahi\b")
_HOWTO = re.compile(r"^\s*(?:how (?:do|can|should|to|would|could)|what (?:should|can|is|are|does)|can i|could i|should i|is it|is there|are there|do you|does your|where (?:can|do)|which)\b")
_HYPO = re.compile(r"\b(?:if|whether|in case|incase|what if|suppose|should it|would it)\b")
_OWN_EVENT = re.compile(r"\bmy (?:account|card|password|profile|login|orders?|details|e-?mail|identity|name|wallet)\b[^.\n]{0,40}\b(?:was|were|has been|have been|got|is being|are being|keeps?|appears?|appeared|shows?|showed|changed|used)\b|"
                        r"\b(?:there (?:is|are|were)|i (?:see|saw|found|noticed|received|got|spotted|have)|i.ve (?:seen|found|noticed|received|got))\b")
_THIRD = re.compile(r"\b(?:friend|colleague|brother|sister|cousin|neighbou?r|relative|someone i know|in the news|article|heard|read about|a customer|"
                    r"my (?:husband|wife|mother|father|dad|mom|son|daughter|uncle|aunt))\b")
_PAY = _c(r"\b(?:payments?|paid|pay|charg(?:e|ed|es|ing)|bill\w*|debit\w*|deduct\w*|transactions?|amount|money|withdr\w*|taken|collected|invoice[sd]?|fees?|statement|"
          r"credit card|debit card|bank|cobr\w*|cargos?|pagos?|pagu\w*|descont\w*|prelev\w*|paiements?|paise|paisa|bhugtan|rupees?)\b|पैसे|भुगतान")
_TWICE = _c(r"\b(?:twice|two times|2 times|2x|double|doubled|duplicat\w*|second time|more than once|multiple times|identical|once more|"
            r"(?:two|2)\s+(?:payments?|charges?|transactions?|deductions?|debits?|withdrawals?|bills?|cobros?|cargos?|pagos?|prelevements?|paiements?|entries|lines|records|listings)|"
            r"(?:two|2)\s+(?:separate\s+)?(?:occasions|times|instances)|do (?:charges?|payments?|debits?|deductions?|transactions?)|"
            r"both\s+(?:payments?|charges?|transactions?|debits?)|dos (?:veces|cobros|cargos|pagos)|deux (?:fois|prelevements|paiements|debits)|en double|doble|do baar|dobara|dubara|2 baar)\b|दो बार|दुबारा")
_NOT_DOUBLE = re.compile(r"double[- ]?check|double[- ]?click|double (?:room|bed|glazing|door|sided)|double the (?:price|cost|amount|quantity|size)|twice as|twice the|"
                         r"twice (?:more|that)|payments? (?:methods?|options?|modes?|ways?|types?|gateways?)|(?:two|2) (?:ways?|options?|methods?|cards?) to pay")
_PAY_HOWTO = re.compile(r"\b(?:can i|could i|may i|how (?:do|can) i|is it possible|do you allow|am i allowed|option to)\b")
_LEGAL = _c(r"\b(?:sue|suing|lawsuits?|legal (?:advice|action|steps|notice|proceedings|counsel|remedy|remedies|recourse)|"
            r"legal (?:representatives?|representation|advis[eo]rs?)|(?:case|claim|complaint|suit|petition)\s+against|lawyers?|attorneys?|solicitors?|advocates?|"
            r"barristers?|courts?|tribunals?|judge|litigation|regulators?|ombudsman|consumer (?:\w+ )?(?:forum|commission|court|protection|agency|authority|council)|"
            r"police|authorities)\b|\b(?:abogad[oa]s?|demand\w*|tribunal\w*|juzgado\w*|denunci\w*|querella|avocats?|poursuit\w*|poursuivre|plainte|justice|vakeel|vakil|kanooni)\b|"
            r"वकील|कानूनी|अदालत|कोर्ट|मुकदमा")
_FUTURE = _c(r"\b(?:will|won.t|going to|gonna|plan(?:ning)? to|intend(?:ing)? to|about to|shall|i.ll|i am|i.m|we.ll|seeking|engaging|hiring|initiating|filing|taking|"
             r"lodging|reporting|expect|hear from|be surprised|if this|unless|otherwise|next step|approach(?:ing)?|could (?:become|end up|go|turn)|may (?:become|end up)|might (?:become|end up))\b|"
             r"\b(?:voy a|hablare|presentare|acudire|acudir|iremos|vamos a|je vais|j.irai|nous allons|karunga|karungi|lunga|lungi|jaunga|jaungi|kar dunga|dalunga|baat karunga)\b|"
             r"करूँगा|करूंगा|जाऊँगा|जाऊंगा")
_LEGAL_NOT = re.compile(r"\b(?:tennis|basketball|squash|food court|courtyard|court shoes?|your (?:legal|compliance) (?:department|team)|terms and conditions|privacy policy|"
                        r"(?:i am|i.m|we are|we.re|as|work as|working as)\s+(?:a|an)\s+(?:lawyers?|attorneys?|solicitors?|advocates?)|"
                        r"(?:lawyers?|attorneys?|solicitors?|legal)[- ](?:approved|friendly|reviewed|certified|checked|age|advice column))\b")
_LEGAL_STRONG = re.compile(r"\b(?:sue|suing|lawsuit|court of law|legal action|legal proceedings|legal notice|take .{0,20} to court|file a case|small claims)\b")
_HIGH = ("account_compromise", "duplicate_payment", "legal_threat")


def _filtered_sentences(name: str, norm: str) -> list[str]:
    """Sentences of `norm` that may legitimately trigger the high-risk intent `name` (exclusions applied)."""
    out = []
    for sent in _SENT.split(norm):
        if not sent.strip():
            continue
        if name == "account_compromise":
            if _HOWTO.match(sent) or _HYPO.search(sent) or (_THIRD.search(sent) and not _OWN_EVENT.search(sent)):
                continue
        elif name == "duplicate_payment":
            sent = _NOT_DOUBLE.sub(" ", sent)
            if _DUP_EXCLUDE.search(sent) or _PAY_HOWTO.search(sent):
                continue
        elif name == "legal_threat":
            if _LEGAL_NOT.search(sent) or _THIRD.search(sent):
                continue
            if _HOWTO.match(sent) and not _LEGAL_STRONG.search(sent):      # "can I speak to a lawyer?" is a question
                continue
        out.append(sent)
    return out


def _concept_intents(norm: str) -> set[str]:
    found: set[str] = set()
    parts = [x for x in _SENT.split(norm) if x.strip()]
    windows = parts + [f"{a}. {b}" for a, b in zip(parts, parts[1:])]        # "A login alert from Brazil. I have never been there."
    for sent in windows:
        skip_ctx = _HOWTO.match(sent) or _HYPO.search(sent) or (_THIRD.search(sent) and not _OWN_EVENT.search(sent))
        if not skip_ctx and ((_ACC_STRONG.search(sent) and _UNAUTH.search(sent))
                             or (_ACC_ACTIVITY.search(sent) and _UNAUTH_STRICT.search(sent))):
            found.add("account_compromise")
        cleaned = _NOT_DOUBLE.sub(" ", sent)
        if (_PAY.search(cleaned) and _TWICE.search(cleaned) and not _DUP_EXCLUDE.search(cleaned)
                and not _PAY_HOWTO.search(cleaned)):
            found.add("duplicate_payment")
        if (_LEGAL.search(sent) and _FUTURE.search(sent) and not _LEGAL_NOT.search(sent) and not _THIRD.search(sent)):
            found.add("legal_threat")
    return found


_VOCAB = ["refund", "charged", "twice", "payment", "duplicate", "account", "hacked", "password", "unauthorized",
          "compromised", "lawyer", "legal", "court", "attorney", "delivery", "delivered", "shipping", "received",
          "damaged", "broken", "working", "delayed", "cancel", "cancelled", "return", "warranty", "policy",
          "urgent", "immediately", "terrible", "useless", "angry", "thanks", "support", "human", "login",
          "blocked", "stolen", "fraud", "deducted", "replacement", "arrived", "double", "billed", "debited",
          "disappointed", "unacceptable", "complaint",
          # real words that must never be 'corrected' into a look-alike (police->policy, changes->charges were real bugs)
          "police", "court", "courts", "hacker", "hackers", "lawyers", "attorneys", "solicitor", "charges", "orders", "accounts",
          "change", "changes", "changed", "charger", "chargers", "chart", "charm", "border", "recorder", "coupon", "coupons"]
_TYPOS = {"twise": "twice", "twics": "twice", "refnd": "refund", "rfund": "refund", "recieved": "received",
          "recived": "received", "acount": "account", "acconut": "account", "hakced": "hacked", "hacekd": "hacked",
          "pasword": "password", "paymnt": "payment", "paymet": "payment", "chargd": "charged", "chraged": "charged",
          "delievery": "delivery", "delivary": "delivery", "oder": "order", "ordr": "order", "lawer": "lawyer",
          "urgnt": "urgent", "imediately": "immediately"}


def normalize(text: str) -> str:
    """Fold, unify apostrophes, collapse letter floods, fix common typos (matching only)."""
    t = fold((text or "").replace("’", "'").replace("‘", "'"))
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)               # heeelp -> help, refunddd -> refund
    out = []
    for word in re.split(r"(\W+)", t):
        if word in _TYPOS:
            word = _TYPOS[word]
        elif len(word) >= 5 and word.isalpha() and word not in _VOCAB:
            close = difflib.get_close_matches(word, _VOCAB, n=1, cutoff=0.88)
            if close:
                word = close[0]
        out.append(word)
    return "".join(out)


@dataclass(frozen=True)
class IntentMatch:
    name: str
    confidence: float


def detect_intents(text: str) -> list[IntentMatch]:
    norm = normalize(text)
    raw = fold(text or "")
    matches: list[IntentMatch] = []
    concept = _concept_intents(norm) | _concept_intents(raw)
    for name, regexes in _COMPILED.items():
        if name in _HIGH:                               # high-risk: exclusions apply to the regex layer as well
            allowed = _filtered_sentences(name, norm) + _filtered_sentences(name, raw)
            hits = sum(1 for rx in regexes if any(rx.search(sn) for sn in allowed))
        else:
            hits = sum(1 for rx in regexes if rx.search(norm) or rx.search(raw))
        if name in concept:
            hits += 1                                   # a concept match is one more independent signal
        if not hits:
            continue
        confidence = min(0.95, 0.7 + 0.12 * (hits - 1))
        if name in ("greeting", "thanks") and len(norm.split()) > 6:
            confidence = 0.55
        matches.append(IntentMatch(name, round(confidence, 2)))
    return sorted(matches, key=lambda m: -m.confidence)


def intent_names(matches: list[IntentMatch], min_confidence: float = 0.0) -> list[str]:
    return [m.name for m in matches if m.confidence >= min_confidence]
