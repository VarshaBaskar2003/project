import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Container, Typography, Button, Paper, Box, Alert, Stack, useTheme,
  LinearProgress, CardMedia, CardContent, Card, Chip, Divider, Grid,
  IconButton, CircularProgress, AlertTitle
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ScienceIcon from "@mui/icons-material/Science";
import RefreshIcon from "@mui/icons-material/Refresh";
import WaterDropIcon from "@mui/icons-material/WaterDrop";
import ThermostatIcon from "@mui/icons-material/Thermostat";
import OpacityIcon from "@mui/icons-material/Opacity";
import TimelineIcon from "@mui/icons-material/Timeline";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import ErrorIcon from "@mui/icons-material/Error";

// API base URL - change this to match your deployment
const API_BASE_URL = "http://127.0.0.1:8000";

const App = () => {
  const theme = useTheme();
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState("");
  const [loading, setLoading] = useState(false);

  // Add states for sensor data
  const [sensorData, setSensorData] = useState(null);
  const [sensorLoading, setSensorLoading] = useState(false);
  const [sensorError, setSensorError] = useState(null);
  const [sensorHistory, setSensorHistory] = useState(null);

  // Add state for webcam capture
  const [captureLoading, setCaptureLoading] = useState(false);
  const [captureError, setCaptureError] = useState(null);

  // Function to fetch sensor data
  const fetchSensorData = async () => {
    setSensorLoading(true);
    setSensorError(null);

    try {
      const response = await axios.get(`${API_BASE_URL}/sensor-data`);
      setSensorData(response.data);
    } catch (err) {
      console.error("Error fetching sensor data:", err);
      setSensorError("Failed to fetch sensor data. Please check if the device is connected.");
    } finally {
      setSensorLoading(false);
    }
  };

  // Function to fetch sensor history
  const fetchSensorHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/sensor-history`);
      setSensorHistory(response.data);
    } catch (err) {
      console.error("Error fetching sensor history:", err);
      // Don't set error here as it's not critical
    }
  };

  // Function to capture image from webcam
  const captureImage = async () => {
    setCaptureLoading(true);
    setCaptureError(null);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE_URL}/capture-image`);

      if (response.data.status === "success") {
        // Create a File object from the captured image URL
        const imageUrl = `${API_BASE_URL}${response.data.image_path}`;

        // Fetch the image as a blob
        const imageResponse = await fetch(imageUrl);
        const blob = await imageResponse.blob();

        // Create a File object
        const fileName = response.data.image_path.split('/').pop();
        const file = new File([blob], fileName, { type: blob.type });

        // Update the state
        setSelectedFile(file);
        setPreview(imageUrl);
        setResult(null);
      } else {
        setCaptureError(response.data.message || "Failed to capture image");
      }
    } catch (err) {
      console.error("Error capturing image:", err);
      setCaptureError(err.response?.data?.message || "Failed to connect to camera. Check if webcam is properly connected.");
    } finally {
      setCaptureLoading(false);
    }
  };

  // Fetch sensor data on component mount and set interval
  useEffect(() => {
    fetchSensorData();
    fetchSensorHistory();

    // Set interval to refresh sensor data every 5 seconds
    const interval = setInterval(() => {
      fetchSensorData();
    }, 5000);

    // Clean up interval on component unmount
    return () => clearInterval(interval);
  }, []);

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict/`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data);
      setError(null);

      // Refresh sensor data after prediction
      fetchSensorData();
    } catch (err) {
      setError(err.response?.data?.detail || "Something went wrong. Please try again.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Function to get color based on moisture level
  const getMoistureColor = (percentage) => {
    if (percentage < 30) return theme.palette.error.main; // Dry
    if (percentage < 70) return theme.palette.warning.main; // Moderate
    return theme.palette.success.main; // Wet
  };

  // Function to get status text based on moisture level
  const getMoistureStatus = (percentage) => {
    if (percentage < 30) return "Dry";
    if (percentage < 70) return "Moderate";
    return "Wet";
  };

  return (
    <Box sx={{ bgcolor: '#f9f9f9', minHeight: '100vh', pt: 2, pb: 4 }}>
      <Container maxWidth="lg">
        <Paper
          elevation={2}
          sx={{
            p: 2,
            mb: 3,
            borderRadius: 2,
            background: theme.palette.primary.main,
            color: 'white',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <Typography
            variant="h4"
            sx={{
              fontWeight: 'bold',
              textAlign: 'center',
            }}
          >
            MaizeCare <Chip label="SensorNet" size="small" sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} />
          </Typography>
        </Paper>

        <Grid container spacing={3}>
          {/* Left column for sensor data */}
          <Grid item xs={12} md={4}>
            <Paper
              elevation={4}
              sx={{
                p: 3,
                borderRadius: 3,
                background: 'linear-gradient(#fff,#e3f2fd)',
                height: '100%'
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" gutterBottom fontWeight="bold" color="primary">
                  Sensor Readings
                </Typography>
                <IconButton
                  size="small"
                  onClick={fetchSensorData}
                  disabled={sensorLoading}
                  color="primary"
                >
                  {sensorLoading ? <CircularProgress size={20} /> : <RefreshIcon />}
                </IconButton>
              </Box>

              {sensorError && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {sensorError}
                </Alert>
              )}

              {sensorData && sensorData.status === "success" ? (
                <Stack spacing={3}>
                  {/* Soil Moisture Card */}
                  <Card variant="outlined" sx={{ borderRadius: 2 }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <WaterDropIcon
                          sx={{
                            color: getMoistureColor(sensorData.data.soil_moisture.percentage),
                            mr: 1
                          }}
                        />
                        <Typography variant="h6" component="div">
                          Soil Moisture
                        </Typography>
                      </Box>

                      <Box sx={{ position: 'relative', pt: 2, display: 'flex', justifyContent: 'center' }}>
                        <CircularProgress
                          variant="determinate"
                          value={sensorData.data.soil_moisture.percentage}
                          size={100}
                          thickness={5}
                          sx={{
                            color: getMoistureColor(sensorData.data.soil_moisture.percentage)
                          }}
                        />
                        <Box
                          sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -20%)',
                            textAlign: 'center'
                          }}
                        >
                          <Typography variant="h5" component="div" fontWeight="bold">
                            {Math.round(sensorData.data.soil_moisture.percentage)}%
                          </Typography>
                          <Typography variant="caption" component="div" color="text.secondary">
                            {sensorData.data.soil_moisture.status}
                          </Typography>
                        </Box>
                      </Box>

                      <Box sx={{ mt: 2, textAlign: 'center' }}>
                        <Chip
                          label={`Raw: ${sensorData.data.soil_moisture.raw}`}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    </CardContent>
                  </Card>

                  {/* Temperature and Humidity - Only show if data is available */}
                  {(sensorData.data.temperature !== undefined || sensorData.data.humidity !== undefined) && (
                    <Card variant="outlined" sx={{ borderRadius: 2 }}>
                      <CardContent>
                        <Typography variant="h6" component="div" sx={{ mb: 2 }}>
                          Environment
                        </Typography>

                        <Grid container spacing={2}>
                          {sensorData.data.temperature !== undefined && (
                            <Grid item xs={6}>
                              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <ThermostatIcon sx={{ color: theme.palette.info.main, mr: 1 }} />
                                <Typography variant="body2">Temperature</Typography>
                              </Box>
                              <Typography variant="h6" fontWeight="bold" color="info.main">
                                {sensorData.data.temperature}°C
                              </Typography>
                            </Grid>
                          )}

                          {sensorData.data.humidity !== undefined && (
                            <Grid item xs={6}>
                              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                <OpacityIcon sx={{ color: theme.palette.info.main, mr: 1 }} />
                                <Typography variant="body2">Humidity</Typography>
                              </Box>
                              <Typography variant="h6" fontWeight="bold" color="info.main">
                                {sensorData.data.humidity}%
                              </Typography>
                            </Grid>
                          )}
                        </Grid>
                      </CardContent>
                    </Card>
                  )}

                  {/* Last Update */}
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="caption" color="text.secondary">
                      Last update: {sensorData.data.last_update}
                    </Typography>
                  </Box>

                  {/* Webcam Capture Button */}
                  <Card variant="outlined" sx={{ borderRadius: 2, mt: 2 }}>
                    <CardContent>
                      <Typography variant="subtitle2" gutterBottom>
                        Quick Capture
                      </Typography>
                      <Button
                        fullWidth
                        variant="contained"
                        color="secondary"
                        startIcon={<CameraAltIcon />}
                        onClick={captureImage}
                        disabled={captureLoading}
                        sx={{ mt: 1 }}
                      >
                        {captureLoading ? 'Capturing...' : 'Capture with Webcam'}
                      </Button>

                      {captureError && (
                        <Alert severity="error" sx={{ mt: 2 }} size="small">
                          {captureError}
                        </Alert>
                      )}
                    </CardContent>
                  </Card>
                </Stack>
              ) : (
                <Box sx={{ textAlign: 'center', py: 5 }}>
                  {sensorLoading ? (
                    <CircularProgress />
                  ) : sensorData && sensorData.status === "waiting" ? (
                    <Alert severity="info">
                      <AlertTitle>Waiting for sensor data</AlertTitle>
                      No data has been received from the sensors yet. Please check the connection to the Arduino.
                    </Alert>
                  ) : (
                    <Typography>No sensor data available</Typography>
                  )}
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Right column for image upload and disease prediction */}
          <Grid item xs={12} md={8}>
            <Paper
              elevation={4}
              sx={{
                p: 4,
                borderRadius: 3,
                background: 'linear-gradient(#fff,#e8f4f8)',
              }}
            >
              <Typography
                variant="h5"
                gutterBottom
                sx={{
                  color: theme.palette.primary.main,
                  fontWeight: 'bold',
                  textAlign: 'center',
                  mb: 3
                }}
              >
                Maize Leaf Disease Analyzer
              </Typography>

              <Box
                component="form"
                onSubmit={handleSubmit}
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center'
                }}
              >
                {!preview ? (
                  <Box
                    sx={{
                      border: `2px dashed ${theme.palette.primary.main}`,
                      borderRadius: 3,
                      p: 4,
                      textAlign: 'center',
                      backgroundColor: theme.palette.background.default,
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        backgroundColor: theme.palette.action.hover,
                      },
                      cursor: 'pointer',
                      mb: 4,
                      maxWidth: 400,
                      width: '100%'
                    }}
                    onClick={() => document.getElementById('upload-button').click()}
                  >
                    <input
                      id="upload-button"
                      type="file"
                      hidden
                      accept="image/*"
                      onChange={handleFileChange}
                    />
                    <CloudUploadIcon sx={{ fontSize: 60, color: theme.palette.primary.main, mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      Drop your maize leaf image here
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      or click to browse files
                    </Typography>
                    <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                      Supported formats: JPG, PNG, JPEG
                    </Typography>
                  </Box>
                ) : (
                  <Card sx={{ mb: 4, overflow: 'hidden', maxWidth: 400, width: '100%' }}>
                    <Box sx={{ position: 'relative' }}>
                      <CardMedia
                        component="img"
                        image={preview}
                        alt="Leaf preview"
                        sx={{
                          height: 300,
                          objectFit: 'cover',
                          width: "100%"
                        }}
                      />
                      <Box
                        sx={{
                          position: 'absolute',
                          bottom: 0,
                          left: 0,
                          right: 0,
                          backgroundColor: 'rgba(0,0,0,0.6)',
                          p: 1,
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }}
                      >
                        <Typography variant="body2" sx={{ color: 'white' }}>
                          {selectedFile?.name}
                        </Typography>
                        <Button
                          size="small"
                          variant="outlined"
                          sx={{ color: 'white', borderColor: 'white' }}
                          onClick={() => {
                            setPreview("");
                            setSelectedFile(null);
                            setResult(null);
                          }}
                        >
                          Change
                        </Button>
                      </Box>
                    </Box>
                  </Card>
                )}

                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  size="large"
                  disabled={!selectedFile || loading}
                  startIcon={<ScienceIcon />}
                  sx={{
                    borderRadius: 2,
                    py: 1,
                    px: 3,
                    width: '100%',
                    maxWidth: 350,
                    fontWeight: 'bold',
                    boxShadow: 3,
                  }}
                >
                  {loading ? 'Analyzing...' : 'Diagnose Leaf'}
                </Button>

                {loading && (
                  <LinearProgress sx={{ mt: 2, borderRadius: 1, width: '100%', maxWidth: 400 }} />
                )}
              </Box>

              {error && (
                <Alert severity="error" sx={{ mt: 3 }} variant="filled">
                  {error}
                </Alert>
              )}

              {result && result.status === "success" && (
                <Box sx={{ mt: 4 }}>
                  <Divider sx={{ my: 2 }}>
                    <Chip
                      icon={<CheckCircleIcon />}
                      label="Analysis Complete"
                      color="success"
                    />
                  </Divider>

                  <Card
                    elevation={2}
                    sx={{
                      mt: 2,
                      borderRadius: 2,
                      border: `1px solid ${theme.palette.divider}`,
                      overflow: 'hidden',
                      maxWidth: 600,
                      mx: 'auto'
                    }}
                  >
                    <Box sx={{
                      p: 2,
                      backgroundColor: theme.palette.primary.main,
                      color: 'white'
                    }}>
                      <Typography variant="h6">Diagnosis Results</Typography>
                    </Box>

                    <CardContent>
                      <Stack spacing={2}>
                        <Box>
                          <Typography variant="subtitle2" color="textSecondary">
                            Detected Disease
                          </Typography>
                          <Typography variant="h5" color="error" fontWeight="bold">
                            {result.disease.replace('_', ' ')}
                          </Typography>
                        </Box>

                        <Box>
                          <Typography variant="subtitle2" color="textSecondary">
                            Confidence Level
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <LinearProgress
                              variant="determinate"
                              value={result.confidence * 100}
                              sx={{
                                flexGrow: 1,
                                height: 10,
                                borderRadius: 5,
                                backgroundColor: theme.palette.grey[300]
                              }}
                            />
                            <Typography variant="h6" fontWeight="bold">
                              {(result.confidence * 100).toFixed(0)}%
                            </Typography>
                          </Box>
                        </Box>

                        <Box>
                          <Typography variant="subtitle2" color="textSecondary">
                            Recommended Action
                          </Typography>
                          <Typography variant="body1" sx={{ p: 2, bgcolor: theme.palette.success.light, borderRadius: 2, color: theme.palette.success.contrastText }}>
                            {result.solution}
                          </Typography>
                        </Box>

                        {/* Display sensor data from result if available */}
                        {result.sensor_data && (
                          <Box>
                            <Typography variant="subtitle2" color="textSecondary">
                              Environmental Conditions
                            </Typography>
                            <Card variant="outlined" sx={{ borderRadius: 2, bgcolor: theme.palette.info.light }}>
                              <CardContent>
                                <Grid container spacing={2}>
                                  <Grid item xs={12} sm={6}>
                                    <Typography variant="body2">
                                      <WaterDropIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                                      Soil Moisture: {Math.round(result.sensor_data.soil_moisture.percentage)}%
                                    </Typography>
                                    <Chip
                                      size="small"
                                      label={getMoistureStatus(result.sensor_data.soil_moisture.percentage)}
                                      sx={{
                                        mt: 1,
                                        bgcolor: getMoistureColor(result.sensor_data.soil_moisture.percentage),
                                        color: '#fff'
                                      }}
                                    />
                                  </Grid>

                                  {result.sensor_data.temperature !== undefined && (
                                    <Grid item xs={12} sm={6}>
                                      <Typography variant="body2">
                                        <ThermostatIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                                        Temperature: {result.sensor_data.temperature}°C
                                      </Typography>
                                      <Chip
                                        size="small"
                                        label={result.sensor_data.temperature > 30 ? "High" : result.sensor_data.temperature < 20 ? "Low" : "Optimal"}
                                        color={result.sensor_data.temperature > 30 ? "warning" : result.sensor_data.temperature < 20 ? "info" : "success"}
                                        sx={{ mt: 1 }}
                                      />
                                    </Grid>
                                  )}

                                  {result.sensor_data.humidity !== undefined && (
                                    <Grid item xs={12} sm={6}>
                                      <Typography variant="body2">
                                        <OpacityIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                                        Humidity: {result.sensor_data.humidity}%
                                      </Typography>
                                      <Chip
                                        size="small"
                                        label={result.sensor_data.humidity > 80 ? "High" : result.sensor_data.humidity < 40 ? "Low" : "Optimal"}
                                        color={result.sensor_data.humidity > 80 ? "warning" : result.sensor_data.humidity < 40 ? "info" : "success"}
                                        sx={{ mt: 1 }}
                                      />
                                    </Grid>
                                  )}
                                </Grid>
                              </CardContent>
                            </Card>
                          </Box>
                        )}

                        {/* Show class confidences */}
                        {result.class_confidences && (
                          <Box>
                            <Typography variant="subtitle2" color="textSecondary">
                              Disease Probability Distribution
                            </Typography>
                            {Object.entries(result.class_confidences).map(([disease, confidence]) => (
                              <Box key={disease} sx={{ mt: 1 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                  <Typography variant="body2">{disease.replace('_', ' ')}</Typography>
                                  <Typography variant="body2" fontWeight="bold">
                                    {(confidence * 100).toFixed(1)}%
                                  </Typography>
                                </Box>
                                <LinearProgress
                                  variant="determinate"
                                  value={confidence * 100}
                                  sx={{
                                    height: 6,
                                    borderRadius: 1,
                                    bgcolor: theme.palette.grey[200],
                                    '& .MuiLinearProgress-bar': {
                                      bgcolor: disease === result.disease ? theme.palette.error.main : theme.palette.primary.main
                                    }
                                  }}
                                />
                              </Box>
                            ))}
                          </Box>
                        )}
                      </Stack>
                    </CardContent>
                  </Card>
                </Box>
              )}

              {result && (result.status === "low_confidence" || result.status === "unknown_disease") && (
                <Alert
                  severity="warning"
                  variant="filled"
                  sx={{ mt: 3 }}
                >
                  <AlertTitle>Low Confidence Detection</AlertTitle>
                  {result.message}
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    {result.recommendation}
                  </Typography>
                </Alert>
              )}

              {result && result.status === "error" && (
                <Alert
                  severity="error"
                  variant="filled"
                  sx={{ mt: 3 }}
                  icon={<ErrorIcon />}
                >
                  <AlertTitle>Processing Error</AlertTitle>
                  {result.message}
                </Alert>
              )}
            </Paper>
          </Grid>

          {/* Sensor History Section - Only show if history is available */}
          {sensorHistory && sensorHistory.status === "success" && (
            <Grid item xs={12}>
              <Paper
                elevation={4}
                sx={{
                  p: 3,
                  borderRadius: 3,
                  background: 'linear-gradient(#fff,#f5f5f5)',
                  mt: 2
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TimelineIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
                  <Typography variant="h6" fontWeight="bold" color="primary">
                    Sensor History
                  </Typography>
                </Box>

                <Box sx={{ overflowX: 'auto' }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', minWidth: 650 }}>
                    {sensorHistory.history.slice().reverse().map((record, index) => (
                      <Card
                        key={index}
                        variant="outlined"
                        sx={{
                          minWidth: 120,
                          m: 1,
                          borderRadius: 2,
                          borderColor: getMoistureColor(record.soil_moisture_percentage)
                        }}
                      >
                        <CardContent sx={{ p: 1, textAlign: 'center' }}>
                          <Typography variant="caption" color="text.secondary" display="block">
                            {record.time.split(' ')[1]}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'center', my: 1 }}>
                            <WaterDropIcon
                              sx={{
                                color: getMoistureColor(record.soil_moisture_percentage),
                                fontSize: 20
                              }}
                            />
                            <Typography variant="h6" fontWeight="bold">
                              {Math.round(record.soil_moisture_percentage)}%
                            </Typography>
                          </Box>

                          {/* Show temperature if available */}
                          {record.temperature !== undefined && (
                            <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                              <ThermostatIcon sx={{ fontSize: 12, verticalAlign: 'middle', mr: 0.5 }} />
                              {record.temperature}°C
                            </Typography>
                          )}

                          {/* Show humidity if available */}
                          {record.humidity !== undefined && (
                            <Typography variant="caption" display="block">
                              <OpacityIcon sx={{ fontSize: 12, verticalAlign: 'middle', mr: 0.5 }} />
                              {record.humidity}%
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                </Box>
              </Paper>
            </Grid>
          )}
        </Grid>
      </Container>
    </Box>
  );
};

export default App;