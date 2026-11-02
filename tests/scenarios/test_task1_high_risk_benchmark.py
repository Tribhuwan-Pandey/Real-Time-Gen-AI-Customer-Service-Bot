"""High-risk intent benchmark: many ways customers describe account compromise, duplicate payment and legal threats
(English, Spanish, French, romanised Hindi, Devanagari), plus innocent look-alikes that must NOT trigger them."""
import pytest

from a_real_time_gen_ai_customer_service_bot.language.intents import detect_intents, intent_names, normalize
from high_risk_sets import SETS

ACCOUNT = [
    "Someone used my card for a purchase I didn't make", "I think my account has been hacked", "There was a login from a device I don't recognize",
    "My password was changed and I didn't do it", "I got a password reset email that I never requested", "Somebody else is using my account",
    "I see unauthorized transactions on my account", "My email address was changed without my permission", "I'm locked out and I think someone took over my account",
    "Fraudulent charge on my card from your site", "Someone logged in as me and placed orders", "My account was breached",
    "There are orders on my account that I did not place", "I received an OTP I didn't request", "My card details were stolen after buying from you",
    "Suspicious activity on my account, I did not do this", "Alguien accedió a mi cuenta sin mi permiso", "Hay cargos que no reconozco en mi cuenta",
    "Me hackearon la cuenta", "Quelqu'un s'est connecté à mon compte", "Je vois des achats que je n'ai pas faits", "Mon compte a été piraté",
    "Mere account mein kisi aur ne login kiya hai", "Mera account hack ho gaya hai", "Mere account se kisi ne order kiya jo maine nahi kiya",
    "मेरे खाते में किसी और ने लॉगिन किया", "मेरा अकाउंट हैक हो गया"]
DUPLICATE = [
    "I was charged twice for the same order", "My card was billed two times", "Payment went through twice", "I paid double by mistake and it was deducted twice",
    "You took the money twice", "There are two identical charges on my statement", "I see a duplicate charge", "Double deduction from my bank account",
    "Charged me again for an order I already paid", "The amount was debited two times", "Money got deducted twice", "Why was my card charged two times for one purchase?",
    "My payment was processed twice", "Me cobraron dos veces", "Veo un cargo duplicado", "Pagué dos veces el mismo pedido", "J'ai été débité deux fois",
    "Je vois un double prélèvement", "Paiement en double sur ma carte", "Mere paise do baar kat gaye", "Payment do baar deduct hua", "दो बार पैसे कट गए"]
LEGAL = [
    "I will take legal action against your company", "My lawyer will contact you", "I'm going to sue you", "I'll file a case in consumer court",
    "I will report this to the consumer protection authority", "You will hear from my attorney", "I'm taking this to court", "I'll send a legal notice",
    "See you in court", "I will complain to the regulator", "I'm consulting a solicitor about this", "I will contact my lawyer today",
    "I'll file a police complaint against you", "This is going to the ombudsman", "Voy a demandarlos", "Hablaré con mi abogado",
    "Presentaré una denuncia ante el tribunal", "Je vais engager des poursuites", "Mon avocat vous contactera", "Je vais saisir le tribunal",
    "Main vakeel se baat karunga", "Main consumer court mein case karunga", "मैं आपके खिलाफ कानूनी कार्रवाई करूँगा"]
INNOCENT = [
    "I have an issue with my order", "Can I pay in two installments?", "How do I change my password?", "I forgot my password, how do I reset it?",
    "How do I log in to my account?", "I want to update my email address", "My order arrived twice, I got two boxes", "What is your refund policy?",
    "Thanks, that was helpful", "Where is my parcel?", "I'd like to pay by card", "Please send me the invoice", "Can you double check my order?",
    "Is my payment confirmed?", "I didn't receive my password reset email", "Can I pay twice a year for the subscription?",
    "My brother is a lawyer and he recommends your products", "I didn't get my order yet", "I did not receive the invoice",
    "Please add a second item to the same order", "Do you accept debit cards?", "I never got the confirmation email"]
HIGH = {"account_compromise": ACCOUNT, "duplicate_payment": DUPLICATE, "legal_threat": LEGAL}


@pytest.mark.parametrize("intent,text", [(i, t) for i, ts in HIGH.items() for t in ts], ids=[f"{i[:4]}:{t[:48]}" for i, ts in HIGH.items() for t in ts])
def test_high_risk_phrasing_is_detected(intent, text):
    assert intent in intent_names(detect_intents(text), 0.5), [(m.name, m.confidence) for m in detect_intents(text)]


@pytest.mark.parametrize("text", INNOCENT, ids=[t[:55] for t in INNOCENT])
def test_innocent_look_alikes_do_not_trigger_high_risk(text):
    assert not set(intent_names(detect_intents(text), 0.5)) & set(HIGH)


def test_recall_summary():
    total = sum(len(t) for t in HIGH.values())
    hits = sum(i in intent_names(detect_intents(t), 0.5) for i, ts in HIGH.items() for t in ts)
    assert hits / total >= 0.95


_POS = [(s, i, t) for s, (pos, _n) in SETS.items() for i, ts in pos.items() for t in ts]
_NEG = [(s, t) for s, (_p, neg) in SETS.items() for t in neg]


@pytest.mark.parametrize("setname,intent,text", _POS, ids=[f"{s}:{i[:4]}:{t[:40]}" for s, i, t in _POS])
def test_holdout_positive(setname, intent, text):
    assert intent in intent_names(detect_intents(text), 0.5), [(m.name, m.confidence) for m in detect_intents(text)]


@pytest.mark.parametrize("setname,text", _NEG, ids=[f"{s}:{t[:48]}" for s, t in _NEG])
def test_holdout_negative(setname, text):
    assert not set(intent_names(detect_intents(text), 0.5)) & set(HIGH)


@pytest.mark.parametrize("typo,expected", [("changes", "changes"), ("charger", "charger"), ("police", "police"), ("chargd", "charged")])
def test_typo_corrector_never_rewrites_real_words_into_risk_words(typo, expected):
    assert normalize(typo) == expected
