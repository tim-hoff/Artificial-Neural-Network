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

(defn gen-matrix
  "generates a `r` by `c` matrix with random weights between -1 and 1."
	[r c & m]
	(for [_ (take r (range))] 
   (for [_ (take c (range))] 
     (* (if (< 0.5 (rand)) -1 1) (rand)))))

(defn l2v
  "convert list to vector matrix"
  [matrix]
  (mapv #(into [] %) matrix))

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
  "takes in `t` and throws it in the sigmoid function\n"
  [t]
  (/ 1 (+ 1 (Math/exp (* -1 t)))))

(defn mmap
  "maps a function on a weight vector matrix"
  [function matrix]
  (mapv #(mapv function %) matrix))

(defn forward
  "takes in weights `w and inputs `x and propagates the inputs though the network"
  [x w]
  (loop [acc x weights w]
    (if (empty? weights)
      acc
      (recur 
         (mmap sigmoid (dot acc (first weights)))
         (rest weights)))))

(defn cost
  "calculates cost"
  [y yhat]
  (* 0.5 (Math/pow (- y yhat) 2)))

(defn mapw
  [function data]
  (mapv #(into [] (function %)) data))

(defn map-data [dataset column fn]
  (i/conj-cols (i/sel dataset :except-cols column)
             (i/$map fn column dataset)))


(defn feed
  "returns adjusted weights, takes in your inputs, weights, and output"
  [x w y]
  (let [yhat (foward x w)]
    
    ))

; test matrixes
(def x  [[1  8.1   6.7   16.1  19    7    1]
         ; [1  8.8   7.7   18.1  20.8  7.4  1]
         ; [1  14.9  13.2  30.1  35.6  12  -1]
         ; [1  15    13.8  31.7  36.9  14  -1]
         ])
(def y (mapv #(vector %) (map peek x)))
(def xx (mapw pop x))
(def nx (norm-scale xx))
(def w (weight-gen `(6 1)))
(def fwd (forward nx w))

(def crab (iio/read-dataset (str (io/resource "crabs.csv")) :header true))
(def crab1 (i/$ [:sp :FL :RW :CL :CW :BD :sex] crab))
(def crab2 (i/to-vect crab1))

(defn scrub 
  "scrubs the first and last attribute, species and gender respectivly"
  [crabs]
  (let [row (into [] (conj (rest crabs) (if (= (first crabs) "B") 0 1)))]
    (into [] (conj (pop row) (if (= (peek row) "F") 0 1)))))

(def crabv (map scrub crab2))

; (i/view crab1)

(println fwd)
(println y)

(defn -main
  "Artificial Neural Networks with stochastic gradient descent optimization"
  [& args]
  (println "done!"))