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
  "takes in `t` and throws it in the sigmoid function\n"
  [t]
  (/ 1 (+ 1 (Math/exp (* -1 t)))))

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

(defn cost
  "calculates cost"
  [y yhat]
  (* 0.5 (Math/pow (- y yhat) 2)))

(defn printfeed
  "prints feed information"
  [x w z lr yhat y xt]
  (println "x") (pm x)
  (println "w") (pm w)
  (println "z") (pm z)
  (println "lr\n" lr)
  (println "yhat") (pm yhat)
  (println "y") (pm y)
  (println "xt") (pm xt)
  )

(defn feed
  "feeds data into nn and returns adjusted weights"
  [x w y lr]
  (let [z (dot x w)
        yhat (mmap sigmoid z)
        xt (transpose x)
        ycost (* -1 (- y yhat))
        
        delta-w (* ycost xt)]
    (printfeed x w z lr yhat y xt) ; print variables for fact checking
    
    ))

(def crab (iio/read-dataset (str (io/resource "crabs.csv")) :header true))
(def crab1 (i/$ [:sp :FL :RW :CL :CW :BD :sex] crab))
(def crab2 (i/to-vect crab1))

(defn scrub 
  "scrubs the first and last attribute, species and gender respectivly"
  [crabs]
  (let [row (into [] (conj (rest crabs) (if (= (first crabs) "B") -1 1)))]
    (into [] (conj (pop row) (if (= (peek row) "F") 0 1)))))

(def crabv (mapv scrub crab2))

; test matrixes
(def x  (into [] (take 1 crabv)))
(def y (mapv #(vector %) (map peek x)))
(def lr (+ 0 0.1))
(def nx (norm-scale (mapw pop x)))
(def w (first (weight-gen `(6 1))))
(feed nx w y lr)

(defn -main
  "Artificial Neural Networks with stochastic gradient descent optimization"
  [& args]
  (println "done!"))