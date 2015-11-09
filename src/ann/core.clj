(ns ann.core
  (:gen-class))

(use 'clojure.repl) ; for doc
(use 'criterium.core) ; benchmarking
(use 'clojure.core.matrix) ; math
(require '[clojure.core.matrix.operators :as mop]); matrix operations
(set-current-implementation :vectorz); matrix computations
(require '[clojure.java.io :as io]) ;io resources
(require '[incanter.core :as i]); statistics library
(require '[incanter.datasets :as ds]) ;  datasets, get-dataset
(require '[incanter.excel :as xls]); excel
(require '[incanter.stats :as stat]); stats
(require '[incanter.charts :as chart]); charts
(require '[incanter.io :as iio]); csv
(set! *warn-on-reflection* true)

(defn l2v
  "convert list to vector matrix"
  [matrix]
  (mapv #(into [] %) matrix))

(defn gen-matrix
  "generates a `r` by `c` matrix with random weights between -1 and 1."
	[r c & m]
	(for [_ (take r (range))] 
   (for [_ (take c (range))] 
     (* (if (< 0.5 (rand)) -1 1) (rand)))))

(defn weight-gen 
  "generates a multitiered matrix"
  [lst]
  (loop [acc (transient []) t lst]
    (if (= 1 (count t))
           (persistent! acc)
           (recur (conj! acc (l2v (apply gen-matrix t))) (drop 1 t)))))

(defn max-fold
  "max values for a matrix"
  [lst]
  (loop [acc (transient []) t lst]
    (if (every? empty? t)
      (rseq (persistent! acc))
      ; (recur (conj! acc (apply max (map peek t))) (map pop t)))))
      (recur (conj! acc (apply max (map abs (map peek t)))) (map pop t))))); handle scaling for negative attributes

(defn norm-scale
  "scales values in matrix to a range between -1 and 1, utilizing max-fld"
  [lst]
  (let [mx (max-fold lst)]
    (mapv #(mapv / % mx) lst)))

(defn sigmoid
  "takes in `z` and throws it in the sigmoid function\n"
  [z]
  (/ 1 (+ 1 (Math/exp (* -1 z)))))

(defn mmap
  "maps a function on a weight vector matrix"
  [function matrix]
  (mapv #(mapv function %) matrix))

(defn forward
  "takes in weights `w and inputs `x and propagates the inputs though the network"
  [input w]
  (loop [x input weights w]
    (if (empty? weights)
      x
      (recur 
         (mmap sigmoid (dot x (first weights))) ; first weights -> weights in this later so
         (rest weights)))))

(defn ppm
  "prints feed information"
  [st info]
  (println st) (pm info))

(defn pluck
  "extract a value from nexted matrix"
  [fn matrix]
  (fn (first matrix)))

(defn printfd
  [x w y lr z yhat xt ycost enz sigmoid-prime delta-w lrdw]
    (ppm "x" x)
    (ppm "w" w)
    (ppm "y" y)
    (ppm "lr" lr)
    (ppm "z" z)
    (ppm "yhat" yhat)
    (ppm "xt" xt)
    (ppm "ycost" ycost)
    (ppm "enz" enz)
    (ppm "sigmoid-prime" sigmoid-prime)
    (ppm "delta-w" delta-w)
    (ppm "lrdw" lrdw)
    (println "..."))

(defn adjust-weights
  "feeds data into nn and returns adjusted weights"
  [x w y lr]
  (let [z (pluck first (dot x w))
        yhat (sigmoid z)
        xt (transpose x); [[x1 x2 x3]] to [[x1] [x2] [x3]]
        ycost (* -1 (- y yhat)); -(y-yhat)
        enz (Math/exp (* -1 z)); e^(-z)
        sigmoid-prime (/ enz (Math/pow (+ 1 enz) 2)); enz/(1+enz)^2
        delta-w (mmap #(* (* ycost sigmoid-prime) %) xt)
        lrdw (mmap #(* % lr) delta-w)
        wkp1 (i/minus w lrdw)]
    ; (printfd x w y lr z yhat xt ycost enz sigmoid-prime delta-w lrdw)
    wkp1
))

(defn feed
  "loops across input and adjustes the weights for all of it. 
  `input` assumes y values are at the end of the vectors"
  [input weight learnrate]
  (loop [x input w weight]
    ; (println "weights - ") (pm w)
    ; (println "x"(count x)"-" (peek x))
    (if (every? empty? x)
      w
      (recur (pop x) (let [thisx (peek x)
                           in [(pop thisx)]
                           out (peek thisx) ] 
                       (adjust-weights in w out learnrate))))))

(defn feed-one 
  [x w]
  (let [z (pluck first (dot x w))
        yhat (sigmoid z)]
  yhat
  ))

(defn finderr
  [value theta match]
  (let [out (if (> value theta) 1.0 0.0)]
    ; (println "value" value "theta" theta "out" out "match" match)
    (if (= out match) 0.0 1.0))
  )

(defn errorcheck
  "checks error given `inputs` `weights` `threshold`"
  [input weight threshold]
  (loop [in input er 0.0]
    (if (every? empty? in)
      er
      (recur 
        (pop in)
        (let [row (peek in)
              x [(pop row)]
              y (peek row)
              yhat (feed-one x weight)]
          (+ er (finderr yhat threshold y)))))))

(defn errloop
  [mn mx step weight ]
  (loop [m mn acc []]
    (if (>= m mx)
      acc
      (recur (+ step m) 
             (conj acc [m (errorcheck crabv w2 m)]))
      )))

(def crab (iio/read-dataset (str (io/resource "crabs.csv")) :header true))
(def crab1 (i/$ [:sp :index :FL :RW :CL :CW :BD :sex] crab))
(def crab2 (i/to-vect crab1))

(defn scrub 
  "scrubs the first and last attribute, species and gender respectivly"
  [crabs]
  (let [row (into [] (conj (rest crabs) (if (= (first crabs) "B") 1.0 -1.0)))]
    (into [] (conj (pop row) (if (= (peek row) "F") 0.0 1.0)))))

(def crabv (norm-scale (mapv scrub crab2)))

(def y (pluck peek crabv))
(def w (first (weight-gen `(7 1))))

(def crabv2 (shuffle (shuffle (into [] (concat 
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                crabv crabv crabv crabv crabv crabv crabv crabv crabv crabv
                                )))))

(def w2 (feed crabv2 w 0.1))
(def w3 (feed crabv2 w2 0.1))
(def w4  (feed crabv2 w3 0.1))

(def wz [[0.252673540652069] [-2.4166590593876145] [3.095142265894501] [-22.964912553504696] [8.910733960205627] [5.117824206109415] [6.579287010853545]])
(println (errorcheck crabv w4 0.5))

(defn -main
  "Artificial Neural Networks with stochastic gradient descent optimization"
  [& args]
  (println "done!"))