// src/components/MainContent.tsx
import React from 'react';
import { Button, Container, Grid, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import '/Users/sharvaysrivastava/WebstormProjects/tsadepression/src/MainContent.css'; // Import the CSS file for transition effects

const MainContent = () => {
  const navigate = useNavigate();

  const handleButtonClick = () => {
    navigate('/getting-started');
  };

  return (
    <Box
      sx={{
        backgroundImage: 'url("/wp2450701.jpg")',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        padding: '50px 0',
        color: '#FFFFFF',
        minHeight: '100vh', // Adjust height to fill the viewport
        display: 'flex',
        alignItems: 'center', // Center content vertically
      }}
    >
      <Container>
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={6}>
            <h1 className="largeHeader">
              Neuropsychiatry powered by machine learning
            </h1>
            <p style={{ fontSize: '1.2rem', color: '#FFFFFF' }}>
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut elit tellus, luctus nec ullamcorper mattis, pulvinar dapibus leo.
            </p>
            <Button
              variant="contained"
              color="secondary"
              size="large"
              onClick={handleButtonClick}
              sx={{ color: '#FFFFFF' }}
            >
              Get Started
            </Button>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box className="coolButton" onClick={handleButtonClick}>
              <Button
                variant="contained"
                color="secondary"
                size="large"
                className="coolButton"
                sx={{ color: '#FFFFFF', borderRadius: '50px' }}
              >
                Click to Start
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default MainContent;
