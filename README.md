# FaceID - Reconocimiento Facial en Android (C++ & OpenCV)

Este proyecto es una aplicación nativa de Android que utiliza algoritmos clásicos de visión por computadora y álgebra lineal para la detección y reconocimiento facial en tiempo real a gran velocidad. Está impulsado por **OpenCV**, procesadores matemáticos compilados en **C++ (NDK)**, e integrado con **Kotlin** y **Java**.

A continuación, se detalla formalmente el funcionamiento técnico de la aplicación, su infraestructura algorítmica y la explicación desglosada por archivos.

---

## 🏗️ Flujo General de Implementación
1. **La Cámara** recibe datos brutos fotográficos 30 veces por segundo.
2. **Java (Android)** inyecta directamente el puntero de la memoria RAM del frame a C++ para evitar estancamientos y copias lentas (`matAddr`).
3. **`FaceDetector`** recibe la imagen, la convierte a escala de grises y extrae solo la porción de píxeles delimitada donde haya un rostro humano.
4. En modo **Entrenamiento**, C++ inyecta aumentos rotacionales a ese rostro capturado e ingresa los puntos a un hiper-plano matemático llamado **PCA (Análisis de Componentes Principales)**.
5. En modo **Reconocimiento**, C++ toma el rostro recortado crudo, lo proyecta matemáticamente al espacio PCA previamente construido, y dictamina a través de la **Distancia Euclidiana** si su margen de error aprueba su entrada como el Propietario.

---

# FaceID - Análisis Lineal y Explicación Exhaustiva de Código

Este documento es una auditoría minuciosa del cómo funciona matemáticamente el código C++ debajo del capó de la aplicación Android. Aquí desglosamos línea por línea qué está haciendo la inteligencia artificial y el procesamiento de imagen en tiempo real para lograr detectar intrusos o permitiendo tu acceso.

---

## 1. El Detector de Rostro: `FaceDetector.cpp`
El paso cero de todo FaceID es ubicar el rostro dentro de la pared, recámara, o bosque en donde estés usando la app. Si tu rostro no se centra y aísla, es imposible analizarte matemáticamente.

```cpp
void FaceDetector::findFacesInImage(Mat &img, Mat &toTest) {
    Mat grayImage;
    // LÍNEA CRÍTICA: Los Haar Cascades (inteligencias previas de búsqueda de sombras) 
    // y el álgebra lineal del PCA no toleran 3 capas de colores (RGB).
    // Esto transforma tu imagen en una estructura bidimensional unicanal de grises.
    cvtColor(img, grayImage, COLOR_RGB2GRAY);
    
    // Nivelador Estabilizador: Fuerza que los blancos sean muy blancos y los negros
    // se acentúen. Ayuda muchísimo si intentas hacer FaceID en tu cuarto oscuro.
    equalizeHist(grayImage, grayImage);

    std::vector<Rect> faces; // Vector donde se guardan rectángulos (rostros detectados)
    
    // Algoritmo Oficial de Viola-Jones:
    // detectMultiScale viaja por los píxeles analizando rectángulos buscando siluetas
    // con sombras que parecen ojos y nariz humana.
    // '1.1' -> Lazo incremental en los barridos multiplicador del 10%.
    // '8' -> Vecinos mínimos requeridos para confirmar que "NO es solo un espejismo".
    // 'Size(150,150)' -> Tamaño mínimo cuadrado en píxeles que debe tener una cara humana.
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 8, 0, Size(150, 150));

    // Si tu cara fue vista en el cuarto...
    if (!faces.empty()) {
        detectedFace = faces[0]; // Captura el bounding box de la principal.
        
        // RECORTADOR (Region Of Interest): Esta técnica extirpa y deja solo
        // el pequeño cuadrado de piel donde estás tú. Eso es "toTest".
        toTest = grayImage(detectedFace).clone(); 
    }
}
```

---

## 2. El Constructor Matemático: `MyPCA.cpp`
PCA (Principal Component Analysis) no mira tu cara como lo hace tu cerebro; la transforma matemáticamente en un plano gigantesco. Su trabajo es agarrar las miles de fotos (`rawData`) de tu entrenamiento y convertirlas en una **Cara Promedio** (Una cara fantasmal que agrupa todas) y extraer tus **Rasgos Fuertes** dominantes (Los `Eigenvectores`).

```cpp
void MyPCA::doPCA() {
  if (rawData.empty()) {
    return; // Parche de seguridad por si no hay caras
  }
  
  // EL APLANAMIENTO MATRICIAL: (RESUMEN EN 1 LÍNEA)
  // 'reshape(1, 1)' toma tu foto cuadrada bidimensional de 150x150
  // y en su lugar forma UNA hiper-lista de 22,500 píxeles lineales larga.
  Mat data = rawData[0].reshape(1, 1).clone();
  
  for (int i = 1; i < rawData.size(); i++) {
    Mat row = rawData[i].reshape(1, 1).clone();
    
    // vconcat une y apila cada cara lineal gigante.
    // Termina formando la "Super Matriz" universal para someterla a cálculos pesados.
    cv::vconcat(data, row, data); 
  }

  // ALGORITMO PURO DE OPENCV: Ejecuta EigenFaces matemáticamente.
  PCA pca(data, Mat(), PCA::DATA_AS_ROW, 0); 
  
  // Construye la MÁSCARA BASE: Tu "rostro modelo" de la media aritmética.
  averageFace = pca.mean.clone(); 
  
  // Componentes Principales: Descifra estadísticamente las curvas y deformidades
  // únicas de tu propia cara, descartando paredes irrelevantes del fondo o brillo.
  eigenvectors = pca.eigenvectors.clone(); 
}
```

---

## 3. Director de Escena y Aumentador: `native-lib.cpp`
Es el puente entre el mundo JNI (Lenguaje de tu celular C y Android) que procesa fotograma por fotograma 30 veces por segundo. Aquí hacemos **Data Augmentation** artificial. Si le dijiste a la App que guardara 10 fotos, el algoritmo las multiplica astutamente a 30 por detrás.

```cpp
// MODO 1: FASE DE ENTRENAMIENTO BIOMÉTRICO (CUANDO PRESIONAS "ENTRENAR")
if (goodFace && !faceMat.empty() && trainingFaces.size() < limitFaces) {
    // 1. Mete tu primera pose intacta cruda y original
    trainingFaces.push_back(faceMat.clone());
    trainingLabels.push_back("Propietario");

    // DATA AUGMENTATION (Inteligencia de Datos Falsos)
    // Para no exigirte tomar 1000 fotos como los celulares IPhone, rotamos tu foto artificialmente aquí:
    
    // a. Crea un compás (Centroide rotacional) de +10° Grados (Inclinación a la derecha).
    Mat rotMat = getRotationMatrix2D(Point2f(faceMat.cols / 2, faceMat.rows / 2), 10, 1.0);
    Mat rotatedFace;
    
    // b. warpAffine rompe y deforma tu matriz basándose en ese compás (aplica la rotación +10).
    warpAffine(faceMat, rotatedFace, rotMat, faceMat.size());
    
    // c. Mete esta foto Falsa como si fuera "otra tuya girada en la vida real".
    trainingFaces.push_back(rotatedFace);
    trainingLabels.push_back("Propietario");

    // Y repite lo mismo para -10° Grados (Inclinación a la izquierda).
    rotMat = getRotationMatrix2D(Point2f(faceMat.cols / 2, faceMat.rows / 2), -10, 1.0);
    // ... lo vuelve a meter a la fuerza a la base de datos de entrenamiento.
}
```

---

## 4. El Juez Interrogador: `FaceRecognizer.cpp`
Aquí está lo bello del álgebra. Cuando llega la cara en vivo tuya (o de un extraño) pidiendo acceso, tú llamas a `euclidean()`. Lo que hace es trazar un "Raycast" numérico midiendo qué tan lejos queda este desconocido extraño, de tus Autovectores sagrados.

```cpp
double FaceRecognizer::euclidean() {
  double dist = 0; // Inicia variable contador numérico.
  
  // Recorre TODOS los miles de pesos numéricos de las características del rostro actual...
  for (int i = 0; i < trainEigen.cols; i++) {
    // Teorema de Pitágoras / Resta Categórica
    // Toma la diferencia literal entre (El vector de prueba o sea el intruso "trainFace")
    // MENOS (El vector original Eigen "trainEigen").
    double d = trainFace.at<float>(0, i) - trainEigen.at<float>(0, i);
    
    // Fórmula oficial L2 Distance:
    // Eleva la diferencia al cuadrado y la suma al tanque general `dist`.
    dist += d * d; 
  }
  return dist; // Finalmente retorna este número enorme y contundente.
}
```

Luego, en el Director **`native-lib.cpp`**, con este dato recién retornado... tomamos rápidamente la importante decisión de la puerta de seguridad matemática:

```cpp
// MODO 2: FASE DE RECONOCEDOR (CUANDO PRESIONAS "RECONOCER")
FaceRecognizer fr(faceMat, myPCA->getAverage(), myPCA->getEigenvectors(),
                  trainData->getFacesInEigen(), trainingLabels);

// Exigimos invocar el comparador
double dist = fr.getClosetDist();

// EL UMBRAL DE SEGURIDAD ABSOLUTO
// Esta es la métrica más valiosa del programa. Si la suma de las diferencias 
// acumuladas calculadas arriba supera el puntaje 2000, automáticamente 
// concluimos que el cráneo de esta persona está muy lejos estadísticamente del tuyo.
if (dist < 2000) { 
    // ÉXITO - Pasaste el margen
    putText(frame, id + " D:" + to_string((int)dist), Point(50, 150),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0, 255), 3);
} else {
    // RECHAZO - El humano no coincide con el Subespacio PCA de la base de datos
    putText(frame, "Desc. D:" + to_string((int)dist), Point(50, 150),
            FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 0, 0, 255), 3);
}
```
