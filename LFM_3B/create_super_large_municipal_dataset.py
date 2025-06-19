#!/usr/bin/env python3
"""
Create a super large German municipal training dataset with 2000+ examples
"""

import json
import random
import itertools


def create_super_large_municipal_dataset(output_file: str = "super_large_municipal_training_data.jsonl"):
    """Create extensive training data with thousands of examples"""
    
    # Massively expanded templates for each department
    super_templates = {
        "einwohnermeldeamt": {
            "questions": [
                # Ummeldung
                "Ich möchte meinen Wohnsitz ummelden. Was muss ich tun?",
                "Wie melde ich meinen Umzug an?",
                "Welche Unterlagen brauche ich für die Ummeldung?",
                "Wo kann ich mich ummelden?",
                "Ich bin umgezogen. Was muss ich beachten?",
                "Wie lange habe ich Zeit für die Ummeldung?",
                "Was kostet die Ummeldung?",
                "Kann ich mich online ummelden?",
                "Ich ziehe ins Ausland. Muss ich mich abmelden?",
                "Wie bekomme ich eine Meldebescheinigung?",
                "Kann mich jemand anderes ummelden?",
                "Was passiert, wenn ich die Ummeldung vergesse?",
                "Muss ich bei einem Umzug innerhalb der Stadt ummelden?",
                "Brauche ich für die Ummeldung einen Termin?",
                "Kann ich mehrere Personen gleichzeitig ummelden?",
                "Was ist eine Wohnungsgeberbestätigung?",
                "Wo bekomme ich die Wohnungsgeberbestätigung?",
                "Ich wohne zur Untermiete. Wie melde ich mich um?",
                "Was mache ich, wenn der Vermieter die Bestätigung verweigert?",
                "Kann ich mich rückwirkend ummelden?",
                # Personalausweis
                "Wie beantrage ich einen neuen Personalausweis?",
                "Was kostet ein Personalausweis?",
                "Wie lange ist ein Personalausweis gültig?",
                "Ich habe meinen Personalausweis verloren. Was nun?",
                "Kann ich einen vorläufigen Personalausweis bekommen?",
                "Welche Unterlagen brauche ich für den Personalausweis?",
                "Wie lange dauert es, bis der Personalausweis fertig ist?",
                "Kann ich den Personalausweis abholen lassen?",
                "Was ist der Unterschied zwischen Personalausweis und Reisepass?",
                "Brauchen Kinder einen Personalausweis?",
                # Meldebescheinigung
                "Was steht in einer Meldebescheinigung?",
                "Wofür brauche ich eine Meldebescheinigung?",
                "Wie lange ist eine Meldebescheinigung gültig?",
                "Kann ich eine Meldebescheinigung online beantragen?",
                "Was ist der Unterschied zwischen einfacher und erweiterter Meldebescheinigung?"
            ],
            "answers": [
                "Für die Ummeldung müssen Sie persönlich beim Einwohnermeldeamt erscheinen. Bringen Sie Ihren Personalausweis oder Reisepass sowie die Wohnungsgeberbestätigung vom Vermieter mit. Die Ummeldung muss innerhalb von 14 Tagen nach dem Umzug erfolgen und ist kostenfrei.",
                "Zur Anmeldung eines neuen Wohnsitzes benötigen Sie einen gültigen Personalausweis oder Reisepass und die Wohnungsgeberbestätigung. Die Anmeldung ist kostenfrei und muss innerhalb von 14 Tagen erfolgen.",
                "Sie benötigen für die Ummeldung: Personalausweis oder Reisepass, Wohnungsgeberbestätigung vom Vermieter oder Eigentümer. Bei minderjährigen Kindern zusätzlich die Geburtsurkunde und bei Sorgerechtsänderungen entsprechende Nachweise.",
                "Die Ummeldung erfolgt im Einwohnermeldeamt oder Bürgerbüro Ihres neuen Wohnortes. Termine können Sie online über das städtische Portal vereinbaren oder telefonisch unter der Nummer 0123-456789.",
                "Nach einem Umzug müssen Sie sich innerhalb von 14 Tagen beim Einwohnermeldeamt ummelden. Vergessen Sie nicht, auch Ihre Krankenversicherung, Bank, Arbeitgeber und die Kfz-Zulassungsstelle über die Adressänderung zu informieren.",
                "Sie haben 14 Tage Zeit, sich nach einem Umzug umzumelden. Bei verspäteter Ummeldung kann ein Bußgeld zwischen 5 und 50 Euro verhängt werden. In Ausnahmefällen kann die Frist verlängert werden.",
                "Die An-, Um- und Abmeldung des Wohnsitzes ist kostenfrei. Lediglich für zusätzliche Bescheinigungen oder Meldebescheinigungen fallen Gebühren von 5 Euro an.",
                "Eine Online-Ummeldung ist derzeit nicht möglich. Sie müssen persönlich beim Einwohnermeldeamt erscheinen. Termine können jedoch online vereinbart werden.",
                "Ja, bei einem dauerhaften Wegzug ins Ausland müssen Sie sich beim Einwohnermeldeamt abmelden. Die Abmeldung ist kostenfrei und Sie erhalten eine Abmeldebescheinigung für Ihre Unterlagen.",
                "Eine Meldebescheinigung erhalten Sie gegen eine Gebühr von 5 Euro beim Einwohnermeldeamt. Sie benötigen dafür Ihren Personalausweis oder Reisepass. Die Bescheinigung wird sofort ausgestellt.",
                "Ja, andere Personen können Sie ummelden, wenn Sie eine schriftliche Vollmacht und eine Kopie Ihres Ausweises mitbringen. Die vollmachtgebende Person muss volljährig sein.",
                "Bei vergessener Ummeldung wird ein Bußgeldverfahren eingeleitet. Die Geldbuße beträgt zwischen 5 und 50 Euro. Melden Sie sich so schnell wie möglich nach, um weitere Konsequenzen zu vermeiden.",
                "Ja, auch bei einem Umzug innerhalb derselben Stadt müssen Sie sich ummelden, wenn sich Ihre Adresse ändert. Die Frist von 14 Tagen gilt auch hier.",
                "Termine sind nicht zwingend erforderlich, aber empfehlenswert, um Wartezeiten zu vermeiden. Termine können Sie online oder telefonisch vereinbaren.",
                "Ja, Sie können mehrere Personen gleichzeitig ummelden, wenn Sie entsprechende Vollmachten und Ausweiskopien aller Personen dabei haben.",
                "Die Wohnungsgeberbestätigung ist ein Formular, das Ihr Vermieter oder Wohnungsgeber ausfüllen muss. Es bestätigt, dass Sie in die Wohnung eingezogen sind.",
                "Die Wohnungsgeberbestätigung erhalten Sie von Ihrem Vermieter, Hausverwaltung oder bei Eigentumswohnungen vom Eigentümer. Ein Blanko-Formular gibt es im Bürgerbüro oder online.",
                "Bei Untermietverhältnissen muss der Hauptmieter die Wohnungsgeberbestätigung ausstellen. Er muss dafür die Erlaubnis des Vermieters zur Untervermietung haben.",
                "Wenn der Vermieter die Bestätigung verweigert, wenden Sie sich an das Einwohnermeldeamt. In begründeten Fällen kann die Behörde auch andere Nachweise akzeptieren.",
                "Eine rückwirkende Ummeldung ist möglich, wird aber mit einem Bußgeld geahndet. Melden Sie sich so schnell wie möglich um und erklären Sie die Verspätung.",
                "Für einen neuen Personalausweis benötigen Sie einen gültigen Nachweis Ihrer Identität, ein aktuelles biometrisches Passfoto und 37 Euro Gebühr. Bei erstmaliger Beantragung zusätzlich eine Geburtsurkunde.",
                "Ein Personalausweis kostet 37 Euro für Personen ab 24 Jahren und 22,80 Euro für Personen unter 24 Jahren. Ein vorläufiger Personalausweis kostet 10 Euro.",
                "Personalausweise sind 10 Jahre gültig für Personen ab 24 Jahren und 6 Jahre für Personen unter 24 Jahren. Die Gültigkeit steht auf der Rückseite des Ausweises.",
                "Bei Verlust des Personalausweises erstatten Sie Anzeige bei der Polizei und beantragen einen neuen beim Bürgerbüro. Der alte Ausweis wird gesperrt.",
                "Einen vorläufigen Personalausweis erhalten Sie sofort für 10 Euro. Er ist 3 Monate gültig und berechtigt nicht zu Reisen ins Ausland.",
                "Sie benötigen: aktuelles biometrisches Passfoto, Nachweis der deutschen Staatsangehörigkeit, bei erstmaliger Beantragung eine Geburtsurkunde.",
                "Die Bearbeitungszeit beträgt etwa 3-4 Wochen. In dringenden Fällen ist ein Express-Service gegen Aufpreis möglich.",
                "Den fertigen Personalausweis können Sie mit einer schriftlichen Vollmacht von einer bevollmächtigten Person abholen lassen.",
                "Der Personalausweis berechtigt nur zu Reisen innerhalb der EU. Für Reisen außerhalb der EU benötigen Sie einen Reisepass.",
                "Kinder unter 16 Jahren können einen Personalausweis beantragen, sind aber nicht dazu verpflichtet. Alternativ reicht oft der Kinderreisepass.",
                "Die Meldebescheinigung enthält Angaben zu Name, Geburtsdatum, aktueller Anschrift und Familienstand. Die erweiterte Variante enthält zusätzlich frühere Anschriften.",
                "Meldebescheinigungen werden häufig von Behörden, Banken, Vermietern oder Arbeitgebern als Nachweis der aktuellen Anschrift verlangt.",
                "Meldebescheinigungen haben keine begrenzte Gültigkeit, sollten aber nicht älter als 3 Monate sein, wenn sie als aktueller Nachweis dienen sollen.",
                "Eine Online-Beantragung von Meldebescheinigungen ist in einigen Gemeinden über das Bürgerportal möglich. Prüfen Sie das Online-Angebot Ihrer Stadt.",
                "Die einfache Meldebescheinigung enthält die aktuelle Anschrift. Die erweiterte zeigt zusätzlich alle Anschriften der letzten 5 Jahre und kostet ebenfalls 5 Euro."
            ]
        },
        
        "bauamt": {
            "questions": [
                # Baugenehmigungen
                "Brauche ich eine Baugenehmigung für einen Wintergarten?",
                "Wie beantrage ich eine Baugenehmigung?",
                "Was kostet eine Baugenehmigung?",
                "Wie lange dauert die Bearbeitung einer Baugenehmigung?",
                "Kann ich ohne Genehmigung ein Gartenhaus bauen?",
                "Welche Unterlagen brauche ich für den Bauantrag?",
                "Wer darf Bauanträge einreichen?",
                "Kann ich meinen Balkon verglassen?",
                "Brauche ich eine Genehmigung für eine Terrasse?",
                "Was ist bei einer Grenzbebauung zu beachten?",
                "Welche Abstände muss ich zum Nachbarn einhalten?",
                "Was ist ein Bauvorantrag?",
                "Kann ich eine Baugenehmigung übertragen?",
                "Wie lange ist eine Baugenehmigung gültig?",
                "Was passiert bei Schwarzbau?",
                # Weitere Bauangelegenheiten
                "Brauche ich eine Genehmigung für eine Garage?",
                "Kann ich mein Dach ausbauen?",
                "Was ist bei einer Dachsanierung zu beachten?",
                "Brauche ich eine Genehmigung für ein Carport?",
                "Kann ich einen Pool im Garten bauen?",
                "Was ist ein Freistellungsverfahren?",
                "Brauche ich eine Genehmigung für Photovoltaik?",
                "Kann ich ein zweites Stockwerk anbauen?",
                "Was kostet ein Bauvorbescheid?",
                "Brauche ich eine Genehmigung für einen Zaun?",
                "Kann ich meine Fassade ändern?",
                "Was ist bei denkmalgeschützten Gebäuden zu beachten?",
                "Brauche ich eine Genehmigung für einen Kamin?"
            ],
            "answers": [
                "Für einen Wintergarten benötigen Sie in der Regel eine Baugenehmigung, da er als Vollbau gilt. Reichen Sie beim Bauamt einen vollständigen Bauantrag mit Bauzeichnungen, Lageplan, Baubeschreibung und statischen Berechnungen ein.",
                "Reichen Sie den Bauantrag mit allen erforderlichen Unterlagen beim Bauamt ein. Der Antrag muss von einem bauvorlageberechtigten Planer eingereicht werden. Die Bearbeitungszeit beträgt etwa 3 Monate.",
                "Die Gebühren für Baugenehmigungen richten sich nach den Baukosten. Als Faustregel gilt: 0,5% bis 1% der Baukosten. Mindestgebühr liegt bei etwa 150 Euro.",
                "Die gesetzliche Bearbeitungszeit beträgt 3 Monate. Bei vollständigen Unterlagen und einfachen Vorhaben kann es auch schneller gehen. Bei komplexen Projekten kann sich die Zeit verlängern.",
                "Gartenhäuser bis 30 Kubikmeter umbauten Raum sind in vielen Bundesländern genehmigungsfrei, müssen aber trotzdem die baurechtlichen Vorschriften wie Abstandsregeln einhalten.",
                "Sie benötigen: Bauzeichnungen (Grundrisse, Schnitte, Ansichten), Lageplan, Baubeschreibung, statische Berechnungen, Nachweis der Erschließung und bei größeren Vorhaben einen Baugrundgutachten.",
                "Bauanträge dürfen nur von bauvorlageberechtigten Personen eingereicht werden. Das sind Architekten, Bauingenieure oder andere qualifizierte Planer mit entsprechender Berechtigung.",
                "Eine Balkonverglasung ist meist genehmigungspflichtig und kann je nach Gebäude und Lage auch wohnungseigentumsrechtliche Zustimmungen erfordern. Informieren Sie sich vorab beim Bauamt.",
                "Größere Terrassen ab einer bestimmten Fläche können genehmigungspflichtig sein. Auch erhöhte Terrassen oder solche mit Überdachung benötigen oft eine Genehmigung.",
                "Bei Grenzbebauung sind die Abstandsregeln der Landesbauordnung zu beachten. Oft ist eine Zustimmung des Nachbarn erforderlich oder es müssen besondere Brandschutzauflagen erfüllt werden.",
                "Die Mindestabstände richten sich nach der Landesbauordnung und betragen meist 2,5 bis 3 Meter. Bei höheren Gebäuden können größere Abstände erforderlich sein.",
                "Ein Bauvorantrag klärt vorab, ob ein Bauvorhaben grundsätzlich genehmigungsfähig ist. Die Gebühr beträgt etwa die Hälfte der späteren Baugenehmigungsgebühr.",
                "Baugenehmigungen sind grundstücksbezogen und können bei Eigentumsübertragung auf den neuen Eigentümer übertragen werden, wenn die Voraussetzungen weiterhin erfüllt sind.",
                "Baugenehmigungen sind in der Regel 3 Jahre gültig. Der Bau muss innerhalb dieser Zeit begonnen werden. Eine Verlängerung ist auf Antrag möglich.",
                "Schwarzbauten müssen meist abgerissen oder nachträglich genehmigt werden. Es können Bußgelder und Zwangsgelder verhängt werden. Wenden Sie sich umgehend an das Bauamt.",
                "Garagen sind meist genehmigungsfrei bis zu einer bestimmten Größe (oft 50 qm), müssen aber die Abstandsregeln einhalten und sind oft anzeigepflichtig.",
                "Ein Dachausbau erfordert meist eine Baugenehmigung, da der Wohnraum vergrößert wird. Prüfen Sie auch die Anforderungen an Wärmeschutz und Brandschutz.",
                "Bei Dachsanierungen sind die Energieeinsparverordnung und eventuelle Denkmalschutzauflagen zu beachten. Größere Änderungen können genehmigungspflichtig sein.",
                "Carports sind oft genehmigungsfrei bis zu einer bestimmten Größe (meist 50 qm), müssen aber die Abstandsregeln einhalten und sind anzeigepflichtig.",
                "Schwimmbecken bis zu einer bestimmten Größe und Tiefe sind meist genehmigungsfrei. Größere Pools können baugenehmigungspflichtig sein. Beachten Sie auch Abstandsregeln.",
                "Im Freistellungsverfahren wird geprüft, ob ein Vorhaben von der Genehmigungspflicht freigestellt werden kann. Das Verfahren ist schneller und kostengünstiger.",
                "Photovoltaikanlagen auf Dächern sind meist genehmigungsfrei, wenn sie die Dachfläche nicht überragen. Bei denkmalgeschützten Gebäuden ist eine Genehmigung erforderlich.",
                "Ein zusätzliches Stockwerk ist ein erheblicher baulicher Eingriff, der immer genehmigungspflichtig ist. Prüfen Sie auch die statischen Voraussetzungen des Gebäudes.",
                "Ein Bauvorbescheid kostet etwa 50% der späteren Baugenehmigungsgebühr. Er klärt vorab rechtliche Fragen und schafft Planungssicherheit.",
                "Zäune bis 1,20 Meter Höhe sind meist genehmigungsfrei, müssen aber Abstandsregeln beachten. Höhere Zäune oder solche an öffentlichen Verkehrsflächen können genehmigungspflichtig sein.",
                "Fassadenänderungen können genehmigungspflichtig sein, besonders bei denkmalgeschützten Gebäuden oder in Gestaltungssatzungsgebieten. Informieren Sie sich vorab.",
                "Bei denkmalgeschützten Gebäuden ist für alle Änderungen eine denkmalrechtliche Genehmigung erforderlich. Wenden Sie sich an die untere Denkmalschutzbehörde.",
                "Kamine und Schornsteine sind meist genehmigungspflichtig. Sie müssen von einem Fachbetrieb geplant und vom Schornsteinfeger abgenommen werden."
            ]
        },
        
        # Continue with other departments...
        "standesamt": {
            "questions": [
                "Wie beantrage ich eine Geburtsurkunde?",
                "Was kostet eine Geburtsurkunde?",
                "Welche Unterlagen brauche ich für die Eheschließung?",
                "Wie lange dauert es, eine Sterbeurkunde zu bekommen?",
                "Kann ich online eine Geburtsurkunde beantragen?",
                "Wo bekomme ich eine beglaubigte Kopie meiner Heiratsurkunde?",
                "Wie ändere ich meinen Namen nach der Heirat?",
                "Was brauche ich für die Anmeldung zur Eheschließung?",
                "Kann ich im Ausland geheiratet haben anerkennen lassen?",
                "Wie beantrage ich ein Lebenspartnerschaftszeugnis?",
                "Was ist eine internationale Geburtsurkunde?",
                "Kann ich eine Eheschließung im Ausland anmelden?",
                "Wie bekomme ich eine mehrsprachige Urkunde?",
                "Was kostet eine Eheschließung?",
                "Kann ich außerhalb des Standesamts heiraten?"
            ],
            "answers": [
                "Geburtsurkunden beantragen Sie beim Standesamt des Geburtsortes. Möglich ist dies persönlich, schriftlich oder in vielen Gemeinden auch online über das Bürgerportal.",
                "Eine Geburtsurkunde kostet 12 Euro. Jede weitere Ausfertigung derselben Urkunde kostet ebenfalls 12 Euro. Für internationale Urkunden fallen 25 Euro an.",
                "Sie benötigen: gültige Ausweisdokumente, beglaubigte Abschrift aus dem Geburtenregister (nicht älter als 6 Monate), aktuelle Aufenthaltsbescheinigung. Bei Geschiedenen zusätzlich das rechtskräftige Scheidungsurteil.",
                "Sterbeurkunden werden in der Regel sofort ausgestellt, wenn alle erforderlichen Unterlagen vorliegen. Die Gebühr beträgt 12 Euro pro Urkunde.",
                "Ja, in vielen Gemeinden können Sie Geburtsurkunden online über das städtische Bürgerportal beantragen. Die Urkunde wird Ihnen dann per Post zugesandt.",
                "Heiratsurkunden erhalten Sie beim Standesamt des Ortes, an dem die Eheschließung stattgefunden hat. Die Gebühr beträgt 12 Euro pro beglaubigter Abschrift.",
                "Nach der Heirat können Sie Ihren Namen beim Einwohnermeldeamt ändern lassen. Bringen Sie dazu Ihre Heiratsurkunde und Ihren Personalausweis mit.",
                "Zur Anmeldung der Eheschließung benötigen Sie gültige Ausweisdokumente, Geburtsurkunden beider Partner und aktuelle Aufenthaltsbescheinigungen.",
                "Ausländische Eheschließungen können beim deutschen Standesamt nachbeurkundet werden, wenn alle Voraussetzungen erfüllt sind. Bringen Sie alle Originaldokumente mit Übersetzungen mit.",
                "Lebenspartnerschaftszeugnisse erhalten Sie beim Standesamt, das die Lebenspartnerschaft begründet hat. Die Gebühr entspricht der für andere Personenstandsurkunden.",
                "Internationale Geburtsurkunden sind mehrsprachige Urkunden, die im Ausland anerkannt werden. Sie kosten 25 Euro und werden vom Geburtsstandesamt ausgestellt.",
                "Eheschließungen im Ausland können Sie beim deutschen Standesamt nachbeurkunden lassen. Dazu müssen die ausländischen Dokumente übersetzt und beglaubigt werden.",
                "Mehrsprachige Urkunden erhalten Sie als internationale Urkunden für 25 Euro oder mit beglaubigten Übersetzungen für zusätzliche Gebühren beim Standesamt.",
                "Die Kosten für eine standesamtliche Trauung betragen etwa 40-80 Euro je nach Gemeinde. Für Trauungen außerhalb der Amtsräume können zusätzliche Gebühren anfallen.",
                "Ja, viele Standesämter bieten auch Trauungen außerhalb der Amtsräume an, z.B. in historischen Gebäuden oder Parks. Dies kostet meist extra."
            ]
        },
        
        # Add more departments with extensive examples...
        "general": {
            "questions": [
                "Wo finde ich Formulare zum Download?",
                "Wie sind die Öffnungszeiten des Rathauses?",
                "Kann ich online einen Termin vereinbaren?",
                "Wo ist das Bürgerbüro?",
                "Wie erreiche ich die Stadtverwaltung?",
                "Gibt es kostenloses WLAN im Rathaus?",
                "Wo kann ich meinen Personalausweis verlängern?",
                "Ist das Rathaus barrierefrei?",
                "Wo finde ich Informationen zu Veranstaltungen?",
                "Wie kann ich der Stadt eine Anregung mitteilen?",
                "Gibt es Parkplätze am Rathaus?",
                "Wo ist die nächste Bushaltestelle?",
                "Kann ich Gebühren mit Karte bezahlen?",
                "Wo bekomme ich einen Stadtplan?",
                "Gibt es eine Bürger-App?"
            ],
            "answers": [
                "Alle städtischen Formulare finden Sie auf der Webseite www.stadt.de unter 'Service/Formulare' oder erhalten sie im Bürgerbüro im Erdgeschoss des Rathauses.",
                "Das Rathaus ist geöffnet: Montag bis Mittwoch 8-16 Uhr, Donnerstag 8-18 Uhr, Freitag 8-12 Uhr. Termine außerhalb der Öffnungszeiten sind nach Vereinbarung möglich.",
                "Ja, Termine können Sie online über das Terminbuchungssystem auf www.stadt.de vereinbaren oder telefonisch unter 0123-456789.",
                "Das Bürgerbüro befindet sich im Erdgeschoss des Rathauses am Haupteingang, Rathausplatz 1. Es ist barrierefrei zugänglich.",
                "Die Stadtverwaltung erreichen Sie telefonisch unter 0123-456789, per E-Mail an info@stadt.de oder persönlich zu den Öffnungszeiten.",
                "Ja, kostenloses WLAN 'Stadt-WLAN' steht im gesamten Rathaus zur Verfügung. Das Passwort erhalten Sie an der Information.",
                "Personalausweise werden im Bürgerbüro im Erdgeschoss des Rathauses verlängert. Termine sind online oder telefonisch buchbar.",
                "Ja, das Rathaus ist vollständig barrierefrei. Es gibt einen Aufzug, behindertengerechte Toiletten und Parkplätze für Menschen mit Behinderung.",
                "Aktuelle Veranstaltungshinweise finden Sie auf der städtischen Webseite, im monatlichen Stadtmagazin und an den Informationstafeln im Rathaus.",
                "Anregungen und Beschwerden können Sie per E-Mail an buergerservice@stadt.de, über das Online-Formular 'Bürgerwünsche' oder persönlich einreichen."
            ]
        }
    }
    
    # Expanded question variations and synonyms
    question_starters = [
        "Wie kann ich", "Wo kann ich", "Was muss ich", "Wann kann ich", "Wie bekomme ich",
        "Wo bekomme ich", "Was brauche ich", "Wie beantrage ich", "Wo beantrage ich",
        "Können Sie mir sagen", "Ich möchte", "Ich brauche", "Ich hätte gerne",
        "Wie funktioniert", "Was kostet", "Wie lange dauert", "Wann ist",
        "Gibt es", "Haben Sie", "Wo finde ich", "Wie finde ich",
        "Ist es möglich", "Kann man", "Darf ich", "Muss ich",
        "Bitte helfen Sie mir", "Ich benötige Hilfe bei", "Könnten Sie mir erklären"
    ]
    
    # Format variations
    format_templates = [
        "Frage: {question}\nAntwort: {answer}",
        "Q: {question}\nA: {answer}",
        "Bürger: {question}\nSachbearbeiter: {answer}",
        "Kunde: {question}\nMitarbeiter: {answer}",
        "Antragsteller: {question}\nBeamter: {answer}",
        "Bürgerin: {question}\nVerwaltung: {answer}",
        "Anfrage: {question}\nAuskunft: {answer}",
        "Problem: {question}\nLösung: {answer}",
        "{question}\n{answer}",
        "Beratungsfall: {question}\nBeratung: {answer}"
    ]
    
    # Generate massive dataset
    all_examples = []
    
    for dept, dept_data in super_templates.items():
        questions = dept_data["questions"]
        answers = dept_data["answers"]
        
        # Original Q&A pairs
        for i, question in enumerate(questions):
            answer = answers[i % len(answers)]  # Cycle through answers
            
            # Multiple formats for each pair
            for template in format_templates:
                text = template.format(question=question, answer=answer)
                all_examples.append({
                    "text": text,
                    "metadata": {"department": dept, "type": "original"}
                })
        
        # Question variations with starters
        for question in questions[:10]:  # First 10 questions per department
            for starter in question_starters[:15]:  # First 15 starters
                if not question.lower().startswith(starter.lower()):
                    # Create variation
                    if starter.endswith(("kann ich", "muss ich", "bekomme ich", "beantrage ich")):
                        # Remove question word from original
                        q_parts = question.split(" ", 2)
                        if len(q_parts) >= 3:
                            new_q = f"{starter} {q_parts[2].lower()}?"
                        else:
                            new_q = f"{starter} {question.lower()}"
                    else:
                        new_q = f"{starter}, {question.lower()}"
                    
                    answer = answers[questions.index(question) % len(answers)]
                    
                    for template in format_templates[:5]:  # Use first 5 templates
                        text = template.format(question=new_q, answer=answer)
                        all_examples.append({
                            "text": text,
                            "metadata": {"department": dept, "type": "variation"}
                        })
    
    # Add conversational patterns
    conversation_patterns = [
        "Guten Tag, {question}",
        "Hallo, {question}",
        "Entschuldigung, {question}",
        "Können Sie mir bitte helfen? {question}",
        "Ich habe eine Frage: {question}",
        "Ich bin unsicher: {question}"
    ]
    
    for pattern in conversation_patterns:
        for dept, dept_data in super_templates.items():
            for question in dept_data["questions"][:5]:  # First 5 per department
                conv_question = pattern.format(question=question)
                answer = dept_data["answers"][dept_data["questions"].index(question) % len(dept_data["answers"])]
                
                for template in format_templates[:3]:
                    text = template.format(question=conv_question, answer=answer)
                    all_examples.append({
                        "text": text,
                        "metadata": {"department": dept, "type": "conversational"}
                    })
    
    # Add incomplete/follow-up questions
    follow_ups = [
        "Und was kostet das?",
        "Wie lange dauert das?",
        "Welche Unterlagen brauche ich dafür?",
        "Kann ich das auch online machen?",
        "Brauche ich einen Termin dafür?",
        "Wo genau muss ich hin?",
        "Kann das jemand anderes für mich machen?",
        "Was passiert, wenn ich das nicht mache?"
    ]
    
    for dept, dept_data in super_templates.items():
        for i, answer in enumerate(dept_data["answers"][:5]):  # First 5 answers
            for follow_up in follow_ups:
                # Extract relevant info from answer for follow-up
                if "kostet" in follow_up.lower() and ("Euro" in answer or "kostenfrei" in answer):
                    follow_answer = "Die Kosten finden Sie in der vorherigen Antwort."
                elif "dauert" in follow_up.lower() and ("Tage" in answer or "Wochen" in answer):
                    follow_answer = "Die Bearbeitungszeit wurde bereits genannt."
                else:
                    follow_answer = "Weitere Details entnehmen Sie bitte der ausführlichen Beratung vor Ort."
                
                text = f"Frage: {follow_up}\nAntwort: {follow_answer}"
                all_examples.append({
                    "text": text,
                    "metadata": {"department": dept, "type": "follow_up"}
                })
    
    # Shuffle all examples
    random.shuffle(all_examples)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(all_examples)} training examples in {output_file}")
    
    # Statistics
    dept_counts = {}
    type_counts = {}
    for ex in all_examples:
        dept = ex["metadata"]["department"]
        etype = ex["metadata"]["type"]
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        type_counts[etype] = type_counts.get(etype, 0) + 1
    
    print(f"\n📊 Super Large Dataset Statistics:")
    print(f"Total examples: {len(all_examples)}")
    print(f"\nBy department:")
    for dept, count in sorted(dept_counts.items()):
        print(f"  {dept}: {count} examples")
    
    print(f"\nBy type:")
    for etype, count in sorted(type_counts.items()):
        print(f"  {etype}: {count} examples")
    
    return output_file


if __name__ == "__main__":
    create_super_large_municipal_dataset()