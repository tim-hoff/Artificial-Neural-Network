(defproject ann "0.1.0-SNAPSHOT"
  :description "Artificial Neural Networks with Stochastic gradient descent"
  :url "http://github.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
 :dependencies [[org.clojure/clojure "1.8.0-beta2"]
                [org.clojure/core.async "0.1.346.0-17112a-alpha"]
  							[criterium "0.4.3"]
                [net.mikera/vectorz-clj "0.36.0"]
								[net.mikera/core.matrix "0.42.1"]
        				[incanter/incanter "1.9.0"]]
  :plugins [[lein-codox "0.9.0"]]
  :main ^:skip-aot ann.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}}
  :jvm-opts ["-Xmx2g" "-server"])
