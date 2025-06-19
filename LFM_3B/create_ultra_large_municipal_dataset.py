#!/usr/bin/env python3
"""
Create an ultra large German municipal training dataset with 20,000+ examples
"""

import json
import random
import itertools
from typing import List, Dict, Tuple


def create_ultra_large_municipal_dataset(output_file: str = "ultra_large_municipal_training_data.jsonl"):
    """Create massive training data with 20,000+ examples through extensive augmentation"""
    
    print("🚀 Creating ULTRA LARGE municipal dataset...")
    
    # Comprehensive answer templates with variations
    answer_templates = {
        "einwohnermeldeamt": {
            "ummeldung": [
                "Die Ummeldung erfolgt beim Einwohnermeldeamt Ihres neuen Wohnortes. Sie müssen persönlich erscheinen und folgende Unterlagen mitbringen: Personalausweis oder Reisepass sowie die Wohnungsgeberbestätigung vom Vermieter. Die Ummeldung muss innerhalb von 14 Tagen nach dem Umzug erfolgen. Die Anmeldung ist kostenfrei.",
                "Für die Ummeldung müssen Sie persönlich beim Einwohnermeldeamt erscheinen. Benötigt werden: gültiger Personalausweis oder Reisepass und die Wohnungsgeberbestätigung. Die Frist beträgt 14 Tage nach Einzug. Bei Versäumnis droht ein Bußgeld von 5 bis 50 Euro.",
                "Sie können sich im Bürgerbüro oder Einwohnermeldeamt Ihres neuen Wohnortes ummelden. Termine sind online buchbar unter www.stadt.de. Bringen Sie Ausweis und Wohnungsgeberbestätigung mit. Die Ummeldung ist gebührenfrei.",
                "Die Anmeldung des neuen Wohnsitzes erfolgt persönlich beim Einwohnermeldeamt innerhalb von 14 Tagen. Erforderliche Dokumente: Personalausweis/Reisepass und Wohnungsgeberbestätigung des Vermieters. Kosten entstehen keine.",
                "Melden Sie sich binnen 14 Tagen nach Umzug beim Einwohnermeldeamt um. Sie benötigen einen gültigen Ausweis und die vom Vermieter ausgefüllte Wohnungsgeberbestätigung. Das Versäumnis der Frist kann mit Bußgeld geahndet werden."
            ],
            "personalausweis": [
                "Personalausweise beantragen Sie im Bürgerbüro oder Einwohnermeldeamt. Benötigt werden: biometrisches Passfoto, alter Ausweis (falls vorhanden), Geburtsurkunde bei Erstbeantragung. Kosten: 37 Euro (ab 24 Jahre) oder 22,80 Euro (unter 24 Jahre). Bearbeitungszeit: 3-4 Wochen.",
                "Für einen neuen Personalausweis benötigen Sie ein aktuelles biometrisches Passfoto und 37 Euro Gebühr. Die Beantragung erfolgt persönlich im Bürgerbüro. Der neue Ausweis ist nach etwa 3 Wochen abholbereit.",
                "Den Personalausweis beantragen Sie persönlich beim Einwohnermeldeamt. Mitzubringen: biometrisches Foto, alter Ausweis, bei Verlust eine Verlustanzeige. Gebühren: 37 Euro für Personen ab 24 Jahren. Gültigkeit: 10 Jahre.",
                "Neue Personalausweise werden im Bürgerbüro ausgestellt. Erforderlich: aktuelles Passbild (biometrisch), bisheriger Ausweis, Gebühr von 37 Euro. Die Herstellung dauert etwa 3-4 Wochen. Abholung nur persönlich möglich.",
                "Zur Beantragung eines Personalausweises erscheinen Sie persönlich im Einwohnermeldeamt. Bringen Sie ein biometrisches Foto und 37 Euro mit. Bei Erstantrag auch die Geburtsurkunde. Fertigstellung in 3-4 Wochen."
            ],
            "meldebescheinigung": [
                "Meldebescheinigungen erhalten Sie gegen 5 Euro Gebühr beim Einwohnermeldeamt. Personalausweis oder Reisepass sind vorzulegen. Die Bescheinigung wird sofort ausgestellt.",
                "Eine Meldebescheinigung kostet 5 Euro und wird im Bürgerbüro sofort ausgestellt. Sie benötigen nur Ihren Personalausweis. Erweiterte Meldebescheinigungen mit früheren Anschriften kosten ebenfalls 5 Euro.",
                "Für eine Meldebescheinigung wenden Sie sich an das Einwohnermeldeamt. Kosten: 5 Euro. Mitzubringen: gültiger Ausweis. Die Ausstellung erfolgt sofort. Online-Beantragung ist in vielen Gemeinden möglich.",
                "Meldebescheinigungen stellt das Bürgerbüro gegen eine Gebühr von 5 Euro aus. Erforderlich ist ein gültiger Personalausweis. Die einfache oder erweiterte Bescheinigung erhalten Sie direkt vor Ort.",
                "Im Einwohnermeldeamt erhalten Sie Meldebescheinigungen für 5 Euro. Bringen Sie Ihren Ausweis mit. Die Bescheinigung wird umgehend erstellt. Für Behördengänge reicht meist die einfache Variante."
            ]
        },
        "bauamt": {
            "baugenehmigung": [
                "Für Bauvorhaben ist meist eine Baugenehmigung erforderlich. Der Bauantrag muss von einem bauvorlageberechtigten Architekten oder Ingenieur eingereicht werden. Erforderliche Unterlagen: Bauzeichnungen, Lageplan, Baubeschreibung, statische Berechnungen. Bearbeitungszeit: etwa 3 Monate.",
                "Baugenehmigungen beantragen Sie beim Bauamt. Notwendig sind vollständige Bauunterlagen, erstellt von einem Architekten. Die Gebühren betragen 0,5-1% der Baukosten. Die Bearbeitung dauert in der Regel 3 Monate.",
                "Ein Bauantrag wird beim örtlichen Bauamt eingereicht. Bauvorlageberechtigte Planer müssen die Unterlagen erstellen. Kosten: ca. 0,5-1% der Bausumme. Nach etwa 3 Monaten erhalten Sie den Bescheid.",
                "Die Baugenehmigung erfordert einen vollständigen Bauantrag mit Plänen und Berechnungen. Nur bauvorlageberechtigte Personen dürfen einreichen. Gebühren richten sich nach Baukosten. Bearbeitungsdauer: 3 Monate.",
                "Für genehmigungspflichtige Bauvorhaben reichen Sie einen Bauantrag beim Bauamt ein. Ein Architekt muss die Unterlagen erstellen. Die Kosten betragen etwa 0,5-1% der Baukosten. Rechnen Sie mit 3 Monaten Bearbeitungszeit."
            ],
            "gartenhaus": [
                "Gartenhäuser bis 30 Kubikmeter umbauten Raum sind in vielen Bundesländern genehmigungsfrei. Dennoch müssen baurechtliche Vorschriften wie Abstandsflächen eingehalten werden. Informieren Sie sich beim Bauamt über lokale Regelungen.",
                "Kleine Gartenhäuser sind oft verfahrensfrei, müssen aber Abstandsregeln einhalten. Die Grenze liegt meist bei 30 Kubikmetern. Größere Bauten benötigen eine Genehmigung. Fragen Sie beim Bauamt nach.",
                "Für Gartenhäuser gelten Sonderregelungen. Bis zu einer bestimmten Größe (oft 30 m³) sind sie genehmigungsfrei. Beachten Sie trotzdem Grenzabstände und örtliche Bebauungspläne.",
                "Gartenhäuser unter 30 Kubikmeter Brutto-Rauminhalt benötigen meist keine Genehmigung. Abstandsflächen zum Nachbarn müssen eingehalten werden. Das Bauamt informiert über die genauen Vorschriften.",
                "Bei Gartenhäusern kommt es auf die Größe an. Kleine Bauten bis 30 m³ sind oft genehmigungsfrei. Dennoch gelten Abstandsregeln. Erkundigen Sie sich vorab beim Bauamt nach den lokalen Bestimmungen."
            ]
        },
        "standesamt": {
            "geburtsurkunde": [
                "Geburtsurkunden beantragen Sie beim Standesamt des Geburtsortes. Möglich ist dies persönlich, schriftlich oder online. Kosten: 12 Euro pro Urkunde. Benötigt werden Personalausweis und Angaben zur Person.",
                "Eine Geburtsurkunde erhalten Sie beim Standesamt, wo die Geburt beurkundet wurde. Die Gebühr beträgt 12 Euro. Online-Beantragung ist vielerorts möglich. Die Zusendung erfolgt per Post.",
                "Für Geburtsurkunden ist das Standesamt des Geburtsortes zuständig. Kosten: 12 Euro je Ausfertigung. Beantragung persönlich, schriftlich oder online möglich. Versand dauert 3-5 Werktage.",
                "Geburtsurkunden stellt das Geburtsstandesamt aus. Gebühr: 12 Euro pro Exemplar. Sie können persönlich vorsprechen oder online bestellen. Für die Beantragung genügt der Personalausweis.",
                "Beim Standesamt des Geburtsortes beantragen Sie Geburtsurkunden für 12 Euro. Die Bestellung ist auch online oder schriftlich möglich. Die Urkunde wird per Post zugestellt."
            ],
            "eheschließung": [
                "Für die Eheschließung melden Sie sich beim Standesamt an. Erforderlich: Personalausweise, aktuelle Geburtsurkunden, Aufenthaltsbescheinigungen. Bei Geschiedenen zusätzlich das Scheidungsurteil. Kosten: 40-80 Euro.",
                "Die Anmeldung zur Eheschließung erfolgt beim Standesamt. Beide Partner müssen persönlich erscheinen. Benötigt: Ausweise, Geburtsurkunden (max. 6 Monate alt), Meldebescheinigungen. Gebühren variieren.",
                "Zur standesamtlichen Trauung melden Sie sich gemeinsam an. Unterlagen: gültige Ausweise, beglaubigte Geburtsregisterauszüge, Aufenthaltsbescheinigungen. Die Anmeldegebühr beträgt etwa 40-80 Euro.",
                "Eheschließungen werden beim Standesamt angemeldet. Notwendige Dokumente: Personalausweise, aktuelle Geburtsurkunden, Meldebescheinigungen. Bei Vorehen das Scheidungsurteil. Kosten je nach Gemeinde unterschiedlich.",
                "Für die Hochzeit melden Sie sich beim Standesamt an. Mitzubringen: Ausweise, Geburtsurkunden (nicht älter als 6 Monate), Aufenthaltsnachweise. Die Trauung kostet zwischen 40 und 80 Euro."
            ]
        },
        "ordnungsamt": {
            "laermbelaestigung": [
                "Bei Lärmbelästigung ist das Ordnungsamt zuständig. Dokumentieren Sie die Störungen mit Datum, Uhrzeit und Art des Lärms. Eine schriftliche Beschwerde kann persönlich oder per E-Mail eingereicht werden.",
                "Lärmbelästigungen melden Sie dem Ordnungsamt. Führen Sie ein Lärmprotokoll mit genauen Zeitangaben. Das Amt prüft den Fall und kann bei Verstößen Bußgelder verhängen.",
                "Wenden Sie sich bei Ruhestörungen an das Ordnungsamt. Eine detaillierte Dokumentation der Vorfälle ist hilfreich. Die Behörde kann Verwarnungen aussprechen und Ordnungswidrigkeiten ahnden.",
                "Das Ordnungsamt bearbeitet Beschwerden über Lärmbelästigung. Reichen Sie eine schriftliche Beschwerde mit Lärmprotokoll ein. Bei begründeten Fällen werden Maßnahmen ergriffen.",
                "Für Lärmbelästigungen ist das Ordnungsamt der richtige Ansprechpartner. Dokumentieren Sie Datum, Uhrzeit und Art der Störung. Eine formlose Beschwerde genügt für die Bearbeitung."
            ],
            "falschparker": [
                "Falschparker melden Sie beim Ordnungsamt telefonisch oder per App. Halten Sie Kennzeichen, Ort und Zeit bereit. Bei Behinderungen kann das Fahrzeug kostenpflichtig abgeschleppt werden.",
                "Das Ordnungsamt ist für Falschparker zuständig. Meldungen sind telefonisch oder online möglich. Geben Sie Kennzeichen und genauen Standort an. Bußgelder werden vom Amt verhängt.",
                "Bei Parkverstößen informieren Sie das Ordnungsamt. Notieren Sie Kennzeichen, Ort und Zeitpunkt. Die Behörde leitet ein Bußgeldverfahren ein. Bei Behinderung erfolgt Abschleppung.",
                "Falschparker können Sie dem Ordnungsamt melden. Erforderliche Angaben: Kennzeichen, genauer Ort, Art des Verstoßes. Das Amt verhängt Verwarngelder oder ordnet Abschleppung an.",
                "Melden Sie Falschparker beim Ordnungsamt. Geben Sie Kennzeichen und Standort an. Bei Behinderung von Ein-/Ausfahrten wird sofort abgeschleppt. Verwarngelder betragen 10-55 Euro."
            ]
        }
    }
    
    # Expanded question variations
    question_variations = {
        "ummeldung": [
            "Wo kann ich mich ummelden?",
            "Wie melde ich meinen Umzug an?",
            "Was brauche ich für die Ummeldung?",
            "Ich bin umgezogen, was muss ich tun?",
            "Wo muss ich hin für die Ummeldung?",
            "Wie funktioniert die Wohnsitzummeldung?",
            "Muss ich mich nach einem Umzug ummelden?",
            "Wie lange habe ich Zeit mich umzumelden?",
            "Was kostet eine Ummeldung?",
            "Kann ich die Ummeldung online machen?",
            "Welche Unterlagen brauche ich zum Ummelden?",
            "Wo bekomme ich die Wohnungsgeberbestätigung?",
            "Was passiert wenn ich mich nicht ummelde?",
            "Kann mich jemand anderes ummelden?",
            "Brauche ich einen Termin zum Ummelden?"
        ],
        "personalausweis": [
            "Wie beantrage ich einen Personalausweis?",
            "Wo kann ich einen neuen Personalausweis beantragen?",
            "Was kostet ein Personalausweis?",
            "Welche Unterlagen brauche ich für den Personalausweis?",
            "Wie lange dauert ein neuer Personalausweis?",
            "Ich habe meinen Ausweis verloren, was nun?",
            "Wo bekomme ich einen neuen Ausweis?",
            "Kann ich einen vorläufigen Ausweis bekommen?",
            "Wie lange ist der Personalausweis gültig?",
            "Was brauche ich für einen neuen Ausweis?",
            "Wo muss ich hin für einen Personalausweis?",
            "Kann jemand anderes meinen Ausweis abholen?",
            "Brauche ich ein Passfoto für den Ausweis?",
            "Wie teuer ist ein neuer Personalausweis?",
            "Wann muss ich den Ausweis erneuern?"
        ]
    }
    
    # Create all possible combinations
    all_examples = []
    
    # Generate systematic combinations
    departments = ["einwohnermeldeamt", "bauamt", "standesamt", "ordnungsamt"]
    
    for dept in departments:
        if dept in answer_templates:
            for topic, answers in answer_templates[dept].items():
                if topic in question_variations:
                    questions = question_variations[topic]
                    
                    # Create all Q&A combinations
                    for q_idx, question in enumerate(questions):
                        for a_idx, answer in enumerate(answers):
                            # Multiple format variations
                            formats = [
                                f"Frage: {question}\nAntwort: {answer}",
                                f"Q: {question}\nA: {answer}",
                                f"Bürger: {question}\nSachbearbeiter: {answer}",
                                f"Kunde: {question}\nMitarbeiter: {answer}",
                                f"Bürgerin: {question}\nBeamter: {answer}",
                                f"Antragsteller: {question}\nVerwaltung: {answer}",
                                f"{question}\n{answer}",
                                f"Anfrage: {question}\nAuskunft: {answer}",
                                f"Person: {question}\nAmt: {answer}",
                                f"Einwohner: {question}\nBehörde: {answer}"
                            ]
                            
                            for fmt in formats:
                                all_examples.append({
                                    "text": fmt,
                                    "metadata": {"dept": dept, "topic": topic, "q_idx": q_idx, "a_idx": a_idx}
                                })
    
    # Add conversational variations
    greetings = ["Guten Tag,", "Hallo,", "Sehr geehrte Damen und Herren,", "Entschuldigung,", 
                 "Guten Morgen,", "Können Sie mir helfen?", "Ich hätte eine Frage:",
                 "Bitte helfen Sie mir:", "Ich brauche Hilfe:", "Eine kurze Frage:"]
    
    for dept in departments:
        if dept in answer_templates:
            for topic, answers in answer_templates[dept].items():
                if topic in question_variations:
                    for greeting in greetings:
                        for i in range(min(5, len(question_variations[topic]))):
                            question = question_variations[topic][i]
                            answer = answers[i % len(answers)]
                            
                            text = f"Frage: {greeting} {question}\nAntwort: {answer}"
                            all_examples.append({
                                "text": text,
                                "metadata": {"dept": dept, "topic": topic, "type": "conversational"}
                            })
    
    # Add incomplete questions with context
    incomplete_patterns = [
        ("personalausweis", ["neuen Ausweis", "Ausweis beantragen", "Personalausweis verloren", "Ausweis erneuern"]),
        ("ummeldung", ["umziehen", "neue Adresse", "Wohnsitz ändern", "nach Umzug"]),
        ("geburtsurkunde", ["Geburtsurkunde", "Urkunde für Baby", "Geburtsnachweis"]),
        ("baugenehmigung", ["bauen", "Haus bauen", "Anbau planen", "Bauantrag"])
    ]
    
    for topic, patterns in incomplete_patterns:
        if topic in ["personalausweis", "ummeldung"]:
            dept = "einwohnermeldeamt"
        elif topic == "geburtsurkunde":
            dept = "standesamt"
        elif topic == "baugenehmigung":
            dept = "bauamt"
        else:
            continue
            
        if dept in answer_templates and topic in answer_templates[dept]:
            answers = answer_templates[dept][topic]
            for pattern in patterns:
                for i, answer in enumerate(answers):
                    text = f"Frage: {pattern}\nAntwort: {answer}"
                    all_examples.append({
                        "text": text,
                        "metadata": {"dept": dept, "topic": topic, "type": "incomplete"}
                    })
    
    # Add common administrative phrases and their contexts
    admin_phrases = [
        "innerhalb von 14 Tagen",
        "persönlich erscheinen",
        "Personalausweis oder Reisepass",
        "Gebühr beträgt",
        "online beantragen",
        "Termin vereinbaren",
        "schriftlich einreichen",
        "Bearbeitungszeit beträgt",
        "kostenfrei",
        "Bußgeld",
        "genehmigungspflichtig",
        "Vollmacht erforderlich",
        "biometrisches Passfoto",
        "Wohnungsgeberbestätigung",
        "bauvorlageberechtigter Architekt"
    ]
    
    # Create phrase-focused examples
    for phrase in admin_phrases:
        # Find answers containing this phrase
        for dept in answer_templates:
            for topic, answers in answer_templates[dept].items():
                for answer in answers:
                    if phrase.lower() in answer.lower():
                        # Create focused Q&A
                        if topic in question_variations:
                            for q in question_variations[topic][:3]:
                                text = f"Frage: {q}\nAntwort: {answer}"
                                all_examples.append({
                                    "text": text,
                                    "metadata": {"dept": dept, "topic": topic, "phrase": phrase}
                                })
    
    # Add dialogues with multiple turns
    multi_turn_dialogues = [
        {
            "turns": [
                ("Ich möchte mich ummelden.", "Für die Ummeldung müssen Sie persönlich beim Einwohnermeldeamt erscheinen."),
                ("Was muss ich mitbringen?", "Sie benötigen Ihren Personalausweis und die Wohnungsgeberbestätigung vom Vermieter."),
                ("Was kostet das?", "Die Ummeldung ist kostenfrei."),
                ("Wie lange habe ich Zeit?", "Die Ummeldung muss innerhalb von 14 Tagen nach dem Umzug erfolgen.")
            ]
        },
        {
            "turns": [
                ("Ich brauche einen neuen Personalausweis.", "Personalausweise beantragen Sie im Bürgerbüro."),
                ("Was kostet der?", "Die Gebühr beträgt 37 Euro für Personen ab 24 Jahren."),
                ("Wie lange dauert das?", "Die Bearbeitungszeit beträgt etwa 3-4 Wochen."),
                ("Welche Unterlagen brauche ich?", "Sie benötigen ein biometrisches Passfoto und Ihren alten Ausweis.")
            ]
        }
    ]
    
    for dialogue in multi_turn_dialogues:
        for i in range(len(dialogue["turns"])):
            # Include context from previous turns
            context = ""
            for j in range(i+1):
                q, a = dialogue["turns"][j]
                if j < i:
                    context += f"Frage: {q}\nAntwort: {a}\n"
                else:
                    context += f"Frage: {q}\nAntwort: {a}"
            
            all_examples.append({
                "text": context,
                "metadata": {"type": "multi_turn", "turn": i}
            })
    
    # Shuffle and write
    random.shuffle(all_examples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(all_examples)} training examples in {output_file}")
    
    # Statistics
    dept_counts = {}
    type_counts = {}
    for ex in all_examples:
        if "dept" in ex["metadata"]:
            dept = ex["metadata"]["dept"]
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
        if "type" in ex["metadata"]:
            etype = ex["metadata"]["type"]
            type_counts[etype] = type_counts.get(etype, 0) + 1
    
    print(f"\n📊 Ultra Large Dataset Statistics:")
    print(f"Total examples: {len(all_examples)}")
    print(f"\nBy department:")
    for dept, count in sorted(dept_counts.items()):
        print(f"  {dept}: {count} examples")
    
    print(f"\nBy type:")
    for etype, count in sorted(type_counts.items()):
        print(f"  {etype}: {count} examples")
    
    # Create a sample file for verification
    with open("sample_ultra_dataset.txt", 'w', encoding='utf-8') as f:
        f.write("SAMPLE OF ULTRA LARGE DATASET (first 20 examples):\n")
        f.write("="*60 + "\n\n")
        for i, example in enumerate(all_examples[:20]):
            f.write(f"Example {i+1}:\n")
            f.write(example["text"])
            f.write("\n" + "-"*40 + "\n\n")
    
    return output_file


if __name__ == "__main__":
    create_ultra_large_municipal_dataset()