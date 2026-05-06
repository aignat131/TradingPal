 ;; ========================================================================
;; 1. DEFINIREA ȘABLOANELOR (TEMPLATES)
;; ========================================================================

(deftemplate stare-sistem
    (slot faza (type SYMBOL) (default initializare))
    (slot obiectiv (type STRING))                    
    (slot strategie (type STRING))                   
    (slot suma-tinta (type FLOAT) (default 0.0))
    (slot perioade-ramase (type INTEGER) (default 1))
    (slot eof (type SYMBOL) (default nu))
    (slot exista-fisier (type SYMBOL) (default da))
)

(deftemplate portofoliu
    (slot buget-disponibil (type FLOAT)) 
)

(deftemplate activ-piata
    (slot nume (type SYMBOL))  
    (slot rsi (type FLOAT))
    (slot pret (type FLOAT))
    (slot ma200 (type FLOAT))
)

(deftemplate detinere-activ
    (slot nume-activ (type SYMBOL))
    (slot cantitate (type FLOAT))
    (slot pret-mediu (type FLOAT))
)

;; ========================================================================
;; 2. BAZA DE REGULI PENTRU FLUX (R1, R2, R3)
;; ========================================================================

(defrule R1-initializare
    ?stare <- (stare-sistem (faza initializare) (exista-fisier da))
    =>
    (printout t "R1: inițializare - Fișier găsit. Trecem la R2." crlf)
    (modify ?stare (faza citire-data))
)

(defrule R2-citire-data-EOF
    ?stare <- (stare-sistem (faza citire-data) (eof da))
    =>
    (printout t "R2: citire dată - S-a atins EOF. Trecem la R3." crlf)
    (modify ?stare (faza finalizare))
)

(defrule R3-finalizare
    (stare-sistem (faza finalizare))
    =>
    (printout t "R3: finalizare - Programul a încheiat ciclul curent." crlf)
)

(defrule Citire-data-Valid
    ?stare <- (stare-sistem (faza citire-data) (eof nu))
    =>
    (printout t "Citire dată - Începem procesarea pe baza obiectivului." crlf)
    (modify ?stare (faza procesare))
)

;; ========================================================================
;; 3. RAMURA: OBIECTIV = INVEST
;; ========================================================================

(defrule Depozitare-fonduri
    (stare-sistem (faza procesare) (obiectiv "Invest"))
    (portofoliu (buget-disponibil ?b&:(<= ?b 0.0)))
    =>
    (printout t "ALERTĂ [Depozitare-fonduri]: Buget insuficient pentru Invest." crlf)
)

(defrule Cumparare-DCA-Fix
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "DCA fix") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?cost (min (/ ?tinta ?perioade) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-DCA-Fix]: DCA Fix. Investim " ?cost "$ în " ?nume "." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-Hibrid-Ajustare-Negativa
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "hibrid") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 20)&:(< ?rsi 40)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?cost (min (* ?baza 1.5) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-Hibrid-Ajustare-Negativa]: Hibrid (RSI " ?rsi "). Piața favorabilă! Investim AGRESIV " ?cost "$ în " ?nume "." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-Hibrid-De-Baza
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "hibrid") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 40)&:(< ?rsi 60)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?cost (min (/ ?tinta ?perioade) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-Hibrid-De-Baza]: Hibrid (RSI " ?rsi "). Piața neutră. Investim NORMAL " ?cost "$ în " ?nume "." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-Hibrid-Ajustare-Pozitiva
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "hibrid") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 60)&:(< ?rsi 80)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?cost (min (* ?baza 0.5) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-Hibrid-Ajustare-Pozitiva]: Hibrid (RSI " ?rsi "). Piața scumpă! Investim DEFENSIV doar " ?cost "$ în " ?nume "." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-RSI-ExtremOversold
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "RSI") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(< ?rsi 30)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?cost (min (* ?baza 3.0) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-RSI-ExtremOversold]: RSI " ?rsi " < 30. Piata EXTREM de ieftina! Cumparam MAXIM " ?cost "$." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-RSI-Oversold
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "RSI") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 30)&:(< ?rsi 45)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?cost (min (* ?baza 2.0) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-RSI-Oversold]: RSI " ?rsi " (30-45). Piata ieftina! Cumparam AGRESIV " ?cost "$." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-RSI-Neutru
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "RSI") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 45)&:(< ?rsi 60)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?cost (min (/ ?tinta ?perioade) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-RSI-Neutru]: RSI " ?rsi " (45-60). Piata neutra. Cumparam NORMAL " ?cost "$." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-RSI-Overbought
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "RSI") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 60)&:(< ?rsi 75)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?cost (min (* ?baza 0.3) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-RSI-Overbought]: RSI " ?rsi " (60-75). Piata scumpa. Cumparam DEFENSIV doar " ?cost "$." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Cumparare-RSI-ExtremOverbought
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Invest") (strategie "RSI") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b&:(> ?b 0.0)))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 75)) (pret ?p) (ma200 ?ma&:(> ?p ?ma)))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c) (pret-mediu ?pm))
    =>
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?cost (min (* ?baza 0.1) ?b))
    (bind ?cant-noua (/ ?cost ?p))
    (bind ?valoare-veche (* ?c ?pm))
    (bind ?valoare-noua (+ ?valoare-veche ?cost))
    (bind ?cant-totala (+ ?c ?cant-noua))
    (bind ?pret-mediu-nou (if (> ?cant-totala 0) then (/ ?valoare-noua ?cant-totala) else 0))

    (printout t "EXEC [Cumparare-RSI-ExtremOverbought]: RSI " ?rsi " >= 75. Piata EXTREM de scumpa! Cumparam MINIMAL " ?cost "$." crlf)

    (modify ?portof (buget-disponibil (- ?b ?cost)))
    (modify ?det (cantitate ?cant-totala) (pret-mediu ?pret-mediu-nou))
    (modify ?stare (suma-tinta (- ?tinta ?cost)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

;; ========================================================================
;; 4. RAMURA: OBIECTIV = CASH OUT
;; ========================================================================

(defrule Lipsa-Active-Pentru-Vanzare
    (stare-sistem (faza procesare) (obiectiv "Cash out"))
    (detinere-activ (cantitate ?c&:(<= ?c 0.0)))
    =>
    (printout t "ALERTĂ [Lipsa-Active-Pentru-Vanzare]: Zero fonduri de vândut." crlf)
)

(defrule Vanzare-DCA
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Cash out") (strategie "DCA fix") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b))
    (activ-piata (nume ?nume) (pret ?p))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c&:(> ?c 0.0)) (pret-mediu ?pm&:(< ?pm ?p)))
    =>
    (bind ?valoare-totala-activ (* ?c ?p))
    (bind ?venit (min (/ ?tinta ?perioade) ?valoare-totala-activ))
    (bind ?cant-vanduta (/ ?venit ?p))

    (printout t "EXEC [Vanzare-DCA]: Vânzare DCA. Extragem " ?venit "$." crlf)

    (modify ?portof (buget-disponibil (+ ?b ?venit)))
    (modify ?det (cantitate (- ?c ?cant-vanduta)))
    (modify ?stare (suma-tinta (- ?tinta ?venit)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Vanzare-Hibrid-Ajustare-Negativă
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Cash out") (strategie "hibrid") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b))
    (activ-piata (nume ?nume) (rsi ?rsi&:(< ?rsi 40)) (pret ?p))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c&:(> ?c 0.0)) (pret-mediu ?pm&:(< ?pm ?p)))
    =>
    (bind ?valoare-totala-activ (* ?c ?p))
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?venit (min (* ?baza 0.2) ?valoare-totala-activ))
    (bind ?cant-vanduta (/ ?venit ?p))

    (printout t "EXEC [Vanzare-Hibrid-Ajustare-Negativa]: Vânzare Hibrid (RSI " ?rsi " < 40). Preț prost. Extragem doar " ?venit "$." crlf)

    (modify ?portof (buget-disponibil (+ ?b ?venit)))
    (modify ?det (cantitate (- ?c ?cant-vanduta)))
    (modify ?stare (suma-tinta (- ?tinta ?venit)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Vanzare-Hibrid-De-Baza
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Cash out") (strategie "hibrid") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 40)&:(< ?rsi 60)) (pret ?p))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c&:(> ?c 0.0)) (pret-mediu ?pm&:(< ?pm ?p)))
    =>
    (bind ?valoare-totala-activ (* ?c ?p))
    (bind ?venit (min (/ ?tinta ?perioade) ?valoare-totala-activ))
    (bind ?cant-vanduta (/ ?venit ?p))

    (printout t "EXEC [Vanzare-Hibrid-De-Baza]: Vânzare Hibrid (RSI " ?rsi "). Neutru. Extragem targetul de " ?venit "$." crlf)

    (modify ?portof (buget-disponibil (+ ?b ?venit)))
    (modify ?det (cantitate (- ?c ?cant-vanduta)))
    (modify ?stare (suma-tinta (- ?tinta ?venit)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Vanzare-Hibrid-Ajustare-Pozitiva
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Cash out") (strategie "hibrid") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 60)&:(< ?rsi 75)) (pret ?p))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c&:(> ?c 0.0)) (pret-mediu ?pm&:(< ?pm ?p)))
    =>
    (bind ?valoare-totala-activ (* ?c ?p))
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?venit (min (* ?baza 1.5) ?valoare-totala-activ))
    (bind ?cant-vanduta (/ ?venit ?p))

    (printout t "EXEC [Vanzare-Hibrid-Ajustare-Pozitiva]: Vânzare Hibrid (RSI " ?rsi "). Trend ascendent. Accelerăm la " ?venit "$ extrași." crlf)

    (modify ?portof (buget-disponibil (+ ?b ?venit)))
    (modify ?det (cantitate (- ?c ?cant-vanduta)))
    (modify ?stare (suma-tinta (- ?tinta ?venit)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Vanzare-RSI-Oportunitate
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Cash out") (strategie "RSI") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 75)) (pret ?p))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c&:(> ?c 0.0)) (pret-mediu ?pm&:(< ?pm ?p)))
    =>
    (bind ?valoare-totala-activ (* ?c ?p))
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?venit (min (* ?baza 2.0) ?valoare-totala-activ))
    (bind ?cant-vanduta (/ ?venit ?p))

    (printout t "EXEC [Vanzare-RSI-Oportunitate]: Vânzare RSI Oportunitate (RSI " ?rsi ")! Cash out MASIV pentru " ?venit "$." crlf)

    (modify ?portof (buget-disponibil (+ ?b ?venit)))
    (modify ?det (cantitate (- ?c ?cant-vanduta)))
    (modify ?stare (suma-tinta (- ?tinta ?venit)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule Vanzare-Hibrid-RSI-Maxim
    ?stare <- (stare-sistem (faza procesare) (obiectiv "Cash out") (strategie "hibrid") (suma-tinta ?tinta) (perioade-ramase ?perioade&:(> ?perioade 0)))
    ?portof <- (portofoliu (buget-disponibil ?b))
    (activ-piata (nume ?nume) (rsi ?rsi&:(>= ?rsi 75)) (pret ?p))
    ?det <- (detinere-activ (nume-activ ?nume) (cantitate ?c&:(> ?c 0.0)) (pret-mediu ?pm&:(< ?pm ?p)))
    =>
    (bind ?valoare-totala-activ (* ?c ?p))
    (bind ?baza (/ ?tinta ?perioade))
    (bind ?venit (min (* ?baza 2.0) ?valoare-totala-activ))
    (bind ?cant-vanduta (/ ?venit ?p))

    (printout t "EXEC [Vanzare-Hibrid-RSI-Maxim]: Vânzare Hibrid (RSI " ?rsi " >= 75). Piata SUPRACUMPARATA! Cash out AGRESIV " ?venit "$." crlf)

    (modify ?portof (buget-disponibil (+ ?b ?venit)))
    (modify ?det (cantitate (- ?c ?cant-vanduta)))
    (modify ?stare (suma-tinta (- ?tinta ?venit)) (perioade-ramase (- ?perioade 1)) (faza finalizare))
)

(defrule fallback-nicio-actiune
    ?stare <- (stare-sistem (faza procesare))
    =>
    (printout t "HOLD [fallback-nicio-actiune]: Nicio regula aplicabila." crlf)
    (modify ?stare (faza finalizare))
)

;; ========================================================================
;; 5. BAZA DE FAPTE PENTRU TESTARE
;; ========================================================================

(deffacts scenariu-test-diagrama
    (stare-sistem 
        (faza initializare) 
        (obiectiv "Invest") 
        (strategie "hibrid")
        (suma-tinta 500.0)
        (perioade-ramase 3)
        (exista-fisier da) 
        (eof nu)
    )
    
    (portofoliu (buget-disponibil 1000.0))
    (activ-piata (nume BTC) (rsi 25.0) (pret 65000.0) (ma200 60000.0))
    (detinere-activ (nume-activ BTC) (cantitate 0.0) (pret-mediu 0.0))
)